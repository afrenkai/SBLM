import argparse
import glob
import math
import os
import time
from dataclasses import asdict
import numpy as np
import torch
import torch._dynamo.config as dynamo_config
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from dion import Dion
from dotenv import load_dotenv
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from loss import BrownianBridgeCriterion
from model import BrownianDDT, DDTConfig
from prep_batch import prepare_diffusion_batch
from sample_and_rollout import noise_from_seeds


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
    ix = torch.randint(0, len(shard) - B * L + 1, (1,)).item()
    input_ids = torch.from_numpy(shard[ix : ix + B * L].astype(np.int64)).to(device).view(B, L)

    # Build cu_seqlens from <|endoftext|> (50256) + row boundaries
    flat = input_ids.view(-1)
    eot_pos = torch.nonzero(flat == 50256, as_tuple=False).squeeze(-1).to(torch.int32)
    row_bounds = torch.arange(0, B * L + 1, L, device=device, dtype=torch.int32)
    cu_seqlens = torch.unique(torch.cat([row_bounds, eot_pos]), sorted=True)

    labels, tau, rope_pos, block_mask = prepare_diffusion_batch(input_ids, cu_seqlens, prefix_prob, logit_norm_mean)
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
    idx = np.random.randint(0, len(imf_seeds), size=(B,))
    x_0 = noise_from_seeds(imf_seeds[idx], L, d_model, device)
    input_ids = torch.from_numpy(imf_tokens[idx].astype(np.int64)).to(device)

    flat = input_ids.view(-1)
    eot_pos = torch.nonzero(flat == 50256, as_tuple=False).squeeze(-1).to(torch.int32)
    row_bounds = torch.arange(0, B * L + 1, L, device=device, dtype=torch.int32)
    cu_seqlens = torch.unique(torch.cat([row_bounds, eot_pos]), sorted=True)

    labels, tau, rope_pos, block_mask = prepare_diffusion_batch(
        input_ids, cu_seqlens, prefix_prob=0.0, logit_norm_mean=logit_norm_mean
    )
    return input_ids, labels, tau, rope_pos, block_mask, x_0


@torch.no_grad()
def evaluate_validation(raw_model: BrownianDDT, val_shards: list[np.memmap], B: int, L: int, device: torch.device):
    was_training = raw_model.training
    raw_model.eval()
    E_weight = raw_model.tok_emb.weight
    val_ce = {}
    encoder_cosines = []
    decoder_ce = LigerCrossEntropyLoss(ignore_index=-100)

    for t_val in (0.2, 0.5, 0.8):
        losses = []
        for validation_batch_index in range(8):
            batch = get_packed_batch(val_shards, B, L, device, prefix_prob=0.0)
            input_ids, rope_pos, block_mask = batch[0], batch[3], batch[4]
            tau = torch.full((B, L), t_val, device=device)
            x_T = F.embedding(input_ids, E_weight).float()
            x_0 = torch.randn_like(x_T)
            eps = torch.randn_like(x_T)
            x_t = (1.0 - t_val) * x_0 + t_val * x_T + 0.3 * math.sqrt(t_val * (1.0 - t_val)) * eps

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = raw_model(x_t, tau, rope_pos, block_mask, labels=None)
                logits, z_t = outputs[0], outputs[1]
            ce = decoder_ce(logits.reshape(-1, logits.size(-1)).clone(), input_ids.reshape(-1))
            losses.append(ce.item())

            if t_val == 0.5:
                t_prime = 0.53
                x_tp = (1.0 - t_prime) * x_0 + t_prime * x_T + 0.3 * math.sqrt(t_prime * (1.0 - t_prime)) * eps
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    z_tp = raw_model.encode(x_tp, torch.full((B, L), t_prime, device=device), rope_pos, block_mask)
                encoder_cosines.append(F.cosine_similarity(z_t.float(), z_tp.float(), dim=-1).mean().item())

        val_ce[t_val] = sum(losses) / len(losses)

    emb_std = E_weight.float().std(dim=0).mean().item()
    sample_rows = F.normalize(E_weight[torch.randperm(E_weight.size(0), device=device)[:1024]].float(), dim=-1)
    off_diag_cos = (sample_rows @ sample_rows.T).fill_diagonal_(0.0).sum().item() / (
        sample_rows.shape[0] * (sample_rows.shape[0] - 1)
    )
    statistics = torch.tensor(
        [val_ce[0.2], val_ce[0.5], val_ce[0.8], emb_std, off_diag_cos, sum(encoder_cosines) / len(encoder_cosines)],
        device=device,
    )
    dist.all_reduce(statistics)
    statistics /= dist.get_world_size()
    values = statistics.cpu().tolist()
    raw_model.train(was_training)
    return {0.2: values[0], 0.5: values[1], 0.8: values[2]}, values[3], values[4], values[5]


def get_lr_scale(step: int, warmup_steps: int, max_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def save_checkpoint(
    model: BrownianDDT,
    opt_dion: Dion,
    opt_adamw: torch.optim.AdamW,
    completed_updates: int,
    args: argparse.Namespace,
    schedule_max_steps: int,
    device: torch.device,
) -> None:
    opt_dion.synchronize_for_checkpoint()
    rank = dist.get_rank()
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "numpy": np.random.get_state(),
    }
    rng_states = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(rng_state, rng_states, dst=0)
    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        path = os.path.join(args.ckpt_dir, f"stage{args.stage}_last.pt")
        torch.save(
            {
                "step": completed_updates,
                "stage": args.stage,
                "model_config": asdict(model.cfg),
                "model": model.state_dict(),
                "optimizer_dion": opt_dion.state_dict(),
                "optimizer_adamw": opt_adamw.state_dict(),
                "training_args": vars(args),
                "schedule_max_steps": schedule_max_steps,
                "rng_states": rng_states,
            },
            path + ".tmp",
        )
        os.replace(path + ".tmp", path)
    dist.barrier()


def main(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.set_float32_matmul_precision("high")
    dynamo_config.recompile_limit = max(128, dynamo_config.recompile_limit)
    torch.manual_seed(1337 + rank)
    np.random.seed(1337 + rank)

    if rank == 0:
        tokens_per_update = args.batch_size * args.seq_len * args.grad_accum * world_size
        print(f"Training on {world_size} GPUs | {tokens_per_update:,} token slots/update", flush=True)

    train_shards = load_bin_shards(os.path.join(args.data_dir, "climbmix_train_*.bin"))
    val_shards = load_bin_shards(os.path.join(args.data_dir, "climbmix_val_*.bin"))

    if args.stage == 2:
        npz_files = sorted(glob.glob(os.path.join(args.imf_dir, "imf_pairs_rank*.npz")))
        loaded = [np.load(f) for f in npz_files]
        imf_seeds = np.concatenate([x["seeds"] for x in loaded], axis=0)
        imf_tokens = np.concatenate([x["tokens"] for x in loaded], axis=0)

    ckpt = None
    checkpoint_path = args.resume_ckpt or args.init_ckpt
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = DDTConfig(**ckpt["model_config"]) if ckpt is not None and "model_config" in ckpt else DDTConfig()
    raw_model = BrownianDDT(cfg).to(device=device)

    if args.init_ckpt:
        raw_model.load_state_dict(ckpt["model"])

    if args.stage == 2 and args.use_qat:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
        from torchao.quantization.qat import QATConfig

        vocab_mods = {raw_model.tok_emb, raw_model.lm_head, raw_model.aux_enc_ce.linear}
        quantize_(
            raw_model,
            QATConfig(Int8DynamicActivationInt8WeightConfig(), step="prepare"),
            filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear) and m not in vocab_mods,
        )

    if args.resume_ckpt:
        assert ckpt["stage"] == args.stage, "Use --init_ckpt when starting a different training stage"
        raw_model.load_state_dict(ckpt["model"])

    compiled_model = torch.compile(raw_model)
    model = DDP(compiled_model, device_ids=[local_rank])
    criterion = BrownianBridgeCriterion(sigma=0.3).to(device)

    groups = raw_model.get_param_groups(dion_lr=args.dion_lr, adam_lr=args.adam_lr)
    opt_dion = Dion(
        groups["dion"],
        replicate_mesh=dist.group.WORLD,
        replicate_mesh_grad_sync=False,
        mu=0.95,
        rank_fraction=0.5,
    )
    opt_adamw = torch.optim.AdamW(groups["adamw"], betas=(0.9, 0.95))

    start_step = 0
    schedule_max_steps = args.max_steps
    if args.resume_ckpt:
        opt_dion.load_state_dict(ckpt["optimizer_dion"])
        opt_adamw.load_state_dict(ckpt["optimizer_adamw"])
        start_step = ckpt["step"]
        schedule_max_steps = ckpt["schedule_max_steps"]
        for argument_name in ("warmup_steps", "dion_lr", "adam_lr"):
            vars(args)[argument_name] = ckpt["training_args"][argument_name]
        assert len(ckpt["rng_states"]) == world_size, "Resume requires the same number of GPUs"
        rng_state = ckpt["rng_states"][rank]
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"], device)
        np.random.set_state(rng_state["numpy"])

    logit_norm_mean = -0.3 if args.stage == 2 else 0.0
    wandb_run = None
    if rank == 0:
        load_dotenv()
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "sblm"),
            entity=os.environ.get("WANDB_ENTITY"),
            id=os.environ.get("WANDB_RUN_ID", os.environ.get("SLURM_JOB_ID")),
            resume="allow",
            settings=wandb.Settings(
                x_disable_stats=True,
                x_disable_meta=True,
                disable_code=True,
                disable_git=True,
                console="off",
            ),
        )
        wandb_run.define_metric("update")
        wandb_run.define_metric("train/*", step_metric="update")
        wandb_run.define_metric("val/*", step_metric="update")
    t0 = time.time()
    window_updates = 0

    for step in range(start_step, args.max_steps):
        lr_mult = get_lr_scale(step, args.warmup_steps, schedule_max_steps)
        for pg in opt_dion.param_groups:
            pg["lr"] = args.dion_lr * lr_mult
        for pg in opt_adamw.param_groups:
            pg["lr"] = args.adam_lr * lr_mult

        opt_dion.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)
        metric_sums = {}

        for micro in range(args.grad_accum):
            model.require_backward_grad_sync = micro == args.grad_accum - 1
            use_imf = (args.stage == 2) and ((step * args.grad_accum + micro) % 2 == 1)

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
            for metric_name, metric_value in metrics.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + metric_value / args.grad_accum

        gradient_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0, error_if_nonfinite=True)
        metric_sums["grad_norm"] = gradient_norm.detach()
        opt_dion.step()
        opt_adamw.step()

        completed_updates = step + 1
        window_updates += 1
        if completed_updates <= 10 or completed_updates % args.log_interval == 0 or completed_updates == args.max_steps:
            metric_names = list(metric_sums)
            metric_values = torch.stack([metric_sums[name] for name in metric_names])
            dist.all_reduce(metric_values)
            metric_values /= world_size
            if rank == 0:
                metric_numbers = metric_values.cpu().tolist()
                reported_metrics = {
                    metric_name: metric_numbers[metric_index] for metric_index, metric_name in enumerate(metric_names)
                }
                assert all(math.isfinite(value) for value in metric_numbers), "Non-finite training metric"
                wandb_run.log(
                    {
                        "update": completed_updates,
                        **{f"train/{name}": value for name, value in reported_metrics.items()},
                        "train/lr_adamw": args.adam_lr * lr_mult,
                        "train/lr_dion": args.dion_lr * lr_mult,
                    }
                )
                dt = time.time() - t0
                toks_per_sec = (
                    args.batch_size * args.seq_len * args.grad_accum * world_size * window_updates / max(dt, 1e-6)
                )
                print(
                    f"step {completed_updates:06d} | loss {reported_metrics['loss']:.4f} | "
                    f"ce {reported_metrics['ce']:.4f} | enc_ce {reported_metrics['enc_ce']:.4f} | "
                    f"dir {reported_metrics['dir']:.4f} | sigreg {reported_metrics['sigreg']:.4f} | "
                    f"cons {reported_metrics['cons']:.4f} | "
                    f"emb_std {reported_metrics['emb_std']:.3f} | grad_norm {reported_metrics['grad_norm']:.3f} | "
                    f"rank0_peak_vram {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB | "
                    f"{toks_per_sec:,.0f} tok/s",
                    flush=True,
                )
                t0 = time.time()
            window_updates = 0

        if completed_updates % args.eval_interval == 0:
            val_ce, emb_std, off_cos, enc_cos = evaluate_validation(
                raw_model, val_shards, args.batch_size, args.seq_len, device
            )
            if rank == 0:
                wandb_run.log(
                    {
                        "update": completed_updates,
                        "val/ce_0.2": val_ce[0.2],
                        "val/ce_0.5": val_ce[0.5],
                        "val/ce_0.8": val_ce[0.8],
                        "val/emb_std": emb_std,
                        "val/vocab_cos": off_cos,
                        "val/encoder_adjacent_cos": enc_cos,
                    }
                )
                print(
                    f"[VAL @ {completed_updates}] CE(0.2)={val_ce[0.2]:.3f} CE(0.5)={val_ce[0.5]:.3f} CE(0.8)={val_ce[0.8]:.3f} | "
                    f"emb_std={emb_std:.3f} | vocab_cos={off_cos:.4f} | enc_adj_cos={enc_cos:.4f}"
                )

        if completed_updates % args.save_interval == 0 or completed_updates == args.max_steps:
            save_checkpoint(raw_model, opt_dion, opt_adamw, completed_updates, args, schedule_max_steps, device)

    opt_dion.synchronize_for_checkpoint()
    dist.barrier()
    dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--data_dir", type=str, default="climbmix_small_bins")
    parser.add_argument("--imf_dir", type=str, default="imf_k1_data")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    checkpoint_options = parser.add_mutually_exclusive_group()
    checkpoint_options.add_argument("--resume_ckpt", type=str, default="")
    checkpoint_options.add_argument("--init_ckpt", type=str, default="")
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
    parser.add_argument("--log_interval", type=int, default=50)
    main(parser.parse_args())
