import argparse
import math
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from model import DDTConfig, BrownianDDT


def noise_from_seeds(seeds: np.ndarray, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """Deterministically reconstructs X_0 ~ N(0, I) of shape (B, L, D) from uint64 seeds."""
    B = len(seeds)
    out = torch.empty((B, seq_len, d_model), device=device, dtype=torch.float32)
    gen = torch.Generator(device=device)
    for i, s in enumerate(seeds):
        gen.manual_seed(int(s))
        out[i].normal_(generator=gen)
    return out


@torch.no_grad()
def sample_brownian_bridge(
    model: BrownianDDT,
    B: int,
    L: int,
    steps: int = 64,
    sigma: float = 0.3,
    cache_interval: int = 2,
    prefix_ids: torch.Tensor | None = None,
    x_0: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Exact-Bridge Euler sampler with DDT Condition Encoder caching.
    """
    E_weight = model.tok_emb.weight
    device = E_weight.device
    D = E_weight.shape[1]

    X = torch.randn((B, L, D), device=device, dtype=torch.float32) if x_0 is None else x_0.float().clone()
    rope_pos = torch.arange(L, device=device, dtype=torch.int32).unsqueeze(0).expand(B, L)

    if prefix_ids is not None:
        P = prefix_ids.shape[1]
        prefix_emb = F.embedding(prefix_ids, E_weight).float()

    z_cache = None
    for k in range(steps):
        t = k / steps
        s = (k + 1) / steps

        if prefix_ids is not None:
            X[:, :P] = prefix_emb

        tau_tensor = torch.full((B, L), t, device=device, dtype=torch.bfloat16)
        if prefix_ids is not None:
            tau_tensor[:, :P] = 1.0

        x_in = X.to(torch.bfloat16)
        if (k % cache_interval == 0) or (z_cache is None):
            z_cache = model.encode(x_in, tau_tensor, rope_pos, block_mask=None)

        logits = model.decode(x_in, tau_tensor, z_cache, rope_pos, block_mask=None)

        if k == steps - 1:
            tokens = logits.argmax(dim=-1)
            if prefix_ids is not None:
                tokens[:, :P] = prefix_ids
            return tokens

        # Clamped endpoint on vocabulary convex hull -> exact conditional bridge transition
        x_hat = torch.softmax(logits.float(), dim=-1) @ E_weight.float()
        alpha = (s - t) / (1.0 - t)
        noise_scale = sigma * math.sqrt((s - t) * (1.0 - s) / (1.0 - t))
        X = (1.0 - alpha) * X + alpha * x_hat + noise_scale * torch.randn_like(X)


def generate_imf_rollouts(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BrownianDDT(DDTConfig()).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(ckpt["model"])
    model.eval()

    per_rank_seqs = args.num_seqs // world_size
    rng = np.random.default_rng(seed=42 + rank)
    all_seeds = rng.integers(0, 2**62, size=(per_rank_seqs,), dtype=np.uint64)
    all_tokens = np.empty((per_rank_seqs, args.seq_len), dtype=np.uint16)

    pbar = tqdm(range(0, per_rank_seqs, args.batch_size), disable=(rank != 0), desc="IMF Rollouts")
    for start in pbar:
        end = min(start + args.batch_size, per_rank_seqs)
        batch_seeds = all_seeds[start:end]
        x_0 = noise_from_seeds(batch_seeds, args.seq_len, model.cfg.d_model, device)
        toks = sample_brownian_bridge(
            model,
            B=len(batch_seeds),
            L=args.seq_len,
            steps=args.steps,
            sigma=args.sigma,
            cache_interval=args.cache_interval,
            x_0=x_0,
        )
        all_tokens[start:end] = toks.cpu().numpy().astype(np.uint16)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"imf_pairs_rank{rank}.npz")
    np.savez(out_path, seeds=all_seeds, tokens=all_tokens)
    if rank == 0:
        print(f"Saved {args.num_seqs:,} coupled IMF pairs across {world_size} ranks to {args.out_dir}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="imf_k1_data")
    parser.add_argument("--num_seqs", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--cache_interval", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.3)
    generate_imf_rollouts(parser.parse_args())
