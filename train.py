import argparse
import glob
import math
import os
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.attention.flex_attention import create_block_mask

from model import DDTConfig, BrownianDDT
from loss import BrownianBridgeCriterion
from sample_and_rollout import noise_from_seeds

# Replace with your local Ampere Dion/Dion2/Dion3 import
from dion import Dion


def load_bin_shards(pattern: str) -> list[np.memmap]:
    paths = sorted(glob.glob(pattern))
    assert len(paths) > 0, f"No .bin files matched {pattern}"
    shards = []
    for p in paths:
        header = np.fromfile(p, count=256, dtype=np.int32)
        assert header[0] == 20240520 and header[1] == 1, f"Invalid header in {p}"
        shards.append(np.memmap(p, dtype=np.uint16, mode="r", offset=256 * 4, shape=(int(header[2]),)))
    return shards


@torch.no_grad()
def get_packed_batch(
    shards: list[np.memmap],
    B: int,
    L: int,
    device: torch.device,
    prefix_prob: float = 0.30,
    logit_norm_mean: float = 0.0,
):
    shard = shards[torch.randint(0, len(shards), (1,)).item()]
    ix = torch.randint(0, len(shard) - B * L - 1, (1,)).item()
    input_ids = torch.from_numpy(shard[ix : ix + B * L].astype(np.int64)).to(device).view(B, L)

    # Build cu_seqlens from <|endoftext|> (50256) + row boundaries
    flat = input_ids.view(-1)
    eot_pos = torch.nonzero(flat == 50256, as_tuple=False).squeeze(-1).to(torch.int32)
    row_bounds = torch.arange(0, B * L + 1, L, device=device, dtype=torch.int32)
    cu_seqlens = torch.unique(torch.cat([row_bounds, eot_pos]), sorted=True)

    positions = torch.arange(B * L, device=device, dtype=torch.int32)
    doc_ids = torch.bucketize(positions, cu_seqlens[1:-1], right=True)
    rope_pos = (positions - cu_seqlens[doc_ids]).view(B, L)

    doc_ids_2d = doc_ids.view(B, L)
    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: doc_ids_2d[b, q_idx] == doc_ids_2d[b, kv_idx],
        B=B, H=None, Q_LEN=L, KV_LEN=L, device=device, _compile=True,
    )

    doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    num_docs = doc_lengths.shape[0]
    tau_docs = torch.sigmoid(torch.randn(num_docs, device=device) + logit_norm_mean)
    tau = tau_docs[doc_ids].view(B, L)

    has_prefix = (torch.rand(num_docs, device=device) < prefix_prob) & (doc_lengths >= 2)
    max_prefix = torch.clamp(doc_lengths // 2, min=1)
    prefix_len_docs = (torch.rand(num_docs, device=device) * max_prefix).long() + 1
    prefix_len_docs = torch.where(has_prefix, prefix_len_docs, torch.zeros_like(prefix_len_docs))

    is_prefix = rope_pos < prefix_len_docs[doc_ids].view(B, L)
    tau = torch.where(is_prefix, torch.ones_like(tau), tau)
    labels = torch.where(is_prefix, torch.full_like(input_ids, -100), input_ids)

    return input_ids, labels, tau, rope_pos, block_mask


@torch.no_grad()
def get_coupled_imf_batch(
    imf_seeds: np.ndarray,
    imf_tokens: np.ndarray,
    B: int,
    L: int,
    d_model: int,
    device: torch.device,
    logit_norm_mean: float = -0.3,
):
    """Loads coupled (X_0, snapped_tokens) pairs for Stage 2 (k=1) IMF distillation."""
    idx = np.random.randint(0, len(imf_seeds), size=(B,))
    x_0 = noise_from_seeds(imf_seeds[idx], L, d_model, device)
    input_ids = torch.from_numpy(imf_tokens[idx].astype(np.int64)).to(device)

    flat = input_ids.view(-1)
    eot_pos = torch.nonzero(flat == 50256, as_tuple=False).squeeze(-1).to(torch.int32)
    row_bounds = torch.arange(0, B * L + 1, L, device=device, dtype=torch.int32)
    cu_seqlens = torch.unique(torch.cat([row_bounds, eot_pos]), sorted=True)

    positions = torch.arange(B * L, device=device, dtype=torch.int32)
    doc_ids = torch.bucketize(positions, cu_seqlens[1:-1], right=True)
    rope_pos = (positions - cu_seqlens[doc_ids]).view(B, L)

    doc_ids_2d = doc_ids.view(B, L)
    block_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: doc_ids_2d[b, q_idx] == doc_ids_2d[b, kv_idx],
        B=B, H=None, Q_LEN=L, KV_LEN=L, device=device, _compile=True,
    )

    doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    tau_docs = torch.sigmoid(torch.randn(doc_lengths.shape[0], device=device) + logit_norm_mean)
    tau = tau_docs[doc_ids].view(B, L)
    return input_ids, input_ids.clone(), tau, rope_pos, block_mask, x_0


@torch.no_grad()
def evaluate_validation(raw_model: BrownianDDT, val_shards: list[np.memmap], B: int, L: int, device: torch.device):
    """Checks all 4 Step-4 sanity criteria on the held-out validation shard."""
    raw_model.eval()
    E_weight = raw_model.tok_emb.weight
    val_ce = {}

    for t_val in (0.2, 0.5, 0.8):
        losses = []
        for _ in range(8):
            input_ids, _, _, rope_pos, block_mask = get_packed_batch(val_shards, B, L, device, prefix_prob=0.0)
            tau = torch.full((B, L), t_val, device=device)
            x_T = F.embedding(input_ids, E_weight).float()
            x_0 = torch.randn_like(x_T)
            eps = torch.randn_like(x_T)
            x_t = ((1.0 - t_val) * x_0 + t_val * x_T + 0.3 * math.sqrt(t_val * (1.0 - t_val)) * eps).to(torch.bfloat16)

            logits, z_t, _ = raw_model(x_t, tau, rope_pos, block_mask, labels=None)
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            losses.append(ce.item())

            if t_val == 0.5:
                # Check encoder smoothness for 2-step caching
                t_prime = 0.53
                x_tp = ((1.0 - t_prime) * x_0 + t_prime * x_T + 0.3 * math.sqrt(t_prime * (1.0 - t_prime)) * eps).to(torch.bfloat16)
                z_tp = raw_model.encode(x_tp, torch.full((B, L), t_prime, device=device), rope_pos, block_mask)
                enc_cos = F.cosine_similarity(z_t.float(), z_tp.float(), dim=-1).mean().item()

        val_ce[t_val] = sum(losses) / len(losses)

    # Check embedding isotropy (std ~ 1.0, pairwise cosine ~ 0.0)
    emb_std = E_weight.float().std(dim=0).mean().item()
    sample_rows = F.normalize(E_weight[torch.randperm(E_weight.size(0), device=device)[:1024]].float(), dim=-1)
    off_diag_cos = (sample_rows @ sample_rows.T).fill_diagonal_(0.0).mean().item()

    raw_model.train()
    return val_ce, emb_std, off_diag_cos, enc_cos


def get_lr_scale(step: int, warmup_steps: int, max_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(1337 + rank)

    train_shards = load_bin_shards(os.path.join(args.data_dir, "climbmix_train_*.bin"))
    val_shards = load_bin_shards(os.path.join(args.data_dir, "climbmix_val_*.bin"))

    if args.stage == 2:
        npz_files = sorted(glob.glob(os.path.join(args.imf_dir, "imf_pairs_rank*.npz")))
        loaded = [np.load(f) for f in npz_files]
        imf_seeds = np.concatenate([x["seeds"] for x in loaded], axis=0)
        imf_tokens = np.concatenate([x["tokens"] for x in loaded], axis=0)

    cfg = DDTConfig()
    raw_model = BrownianDDT(cfg).to(device=device, dtype=torch.bfloat16)

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])

    # Optional Stage 2 INT8 QAT (keeps embedding & lm_head in BF16)
    if args.stage == 2 and args.use_qat:
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        from torchao.quantization.qat import QATConfig
        vocab_mods = {raw_model.tok_emb, raw_model.lm_head, raw_model.aux_enc_ce}
        quantize_(
            raw_model,
            QATConfig(Int8DynamicActivationInt8WeightConfig(), step="prepare"),
            filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and m not in vocab_mods,
        )

    compiled_model = torch.compile(raw_model)
    model = DDP(compiled_model, device_ids=[local_rank])
    criterion = BrownianBridgeCriterion(sigma=0.3).to(device)

    groups = raw_model.get_param_groups(dion_lr=args.dion_lr, adam_lr=args.adam_lr)
    opt_dion = Dion(groups["dion"], mu=0.95, rank_fraction=0.5)
    opt_adamw = torch.optim.AdamW(groups["adamw"], betas=(0.9, 0.95))

    logit_norm_mean = -0.3 if args.stage == 2 else 0.0
    t0 = time.time()

    for step in range(args.max_steps):
        lr_mult = get_lr_scale(step, args.warmup_steps, args.max_steps)
        for pg in opt_dion.param_groups:
            pg["lr"] = args.dion_lr * lr_mult
        for pg in opt_adamw.param_groups:
            pg["lr"] = args.adam_lr * lr_mult

        opt_dion.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        for micro in range(args.grad_accum):
            model.require_backward_grad_sync = (micro == args.grad_accum - 1)
            use_imf = (args.stage == 2) and (micro % 2 == 1)

            if use_imf:
                input_ids, labels, tau, rope_pos, block_mask, x_0 = get_coupled_imf_batch(
                    imf_seeds, imf_tokens, args.batch_size, args.seq_len, cfg.d_model, device, logit_norm_mean
                )
            else:
                input_ids, labels, tau, rope_pos, block_mask = get_packed_batch(
                    train_shards, args.batch_size, args.seq_len, device, logit_norm_mean=logit_norm_mean
                )
                x_0 = None

            loss, metrics = criterion(
                model, raw_model, input_ids, labels, tau, rope_pos, block_mask, step, x_0_override=x_0
            )
            (loss / args.grad_accum).backward()

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        opt_dion.step()
        opt_adamw.step()

        if rank == 0 and step % 50 == 0:
            dt = time.time() - t0
            toks_per_sec = (args.batch_size * args.seq_len * args.grad_accum * world_size * 50) / max(dt, 1e-6)
            t0 = time.time()
            print(
                f"step {step:06d} | loss {metrics['loss']:.4f} | ce {metrics['ce']:.4f} | "
                f"enc_ce {metrics['enc_ce']:.4f} | dir {metrics['dir']:.4f} | "
                f"sigreg {metrics['sigreg']:.4f} | emb_std {metrics['emb_std']:.3f} | {toks_per_sec:,.0f} tok/s"
            )

        if step > 0 and step % args.eval_interval == 0:
            val_ce, emb_std, off_cos, enc_cos = evaluate_validation(
                raw_model, val_shards, args.batch_size, args.seq_len, device
            )
            if rank == 0:
                print(
                    f"[VAL @ {step}] CE(0.2)={val_ce[0.2]:.3f} CE(0.5)={val_ce[0.5]:.3f} CE(0.8)={val_ce[0.8]:.3f} | "
                    f"emb_std={emb_std:.3f} | vocab_cos={off_cos:.4f} | enc_adj_cos={enc_cos:.4f}"
                )

        if step > 0 and step % args.save_interval == 0:
            if hasattr(opt_dion, "synchronize_for_checkpoint"):
                opt_dion.synchronize_for_checkpoint()
            if rank == 0:
                os.makedirs(args.ckpt_dir, exist_ok=True)
                torch.save(
                    {"step": step, "model": raw_model.state_dict()},
                    os.path.join(args.ckpt_dir, f"stage{args.stage}_step{step:06d}.pt"),
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--data_dir", type=str, default="climbmix_small_bins")
    parser.add_argument("--imf_dir", type=str, default="imf_k1_data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--use_qat", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=35000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--dion_lr", type=float, default=1.5e-3)
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=5000)
    main(parser.parse_args())
