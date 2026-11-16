import torch

def precompute_rope_freqs(dim: int, max_seq_len: int = 2048, theta: float = 10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    return torch.cos(angles).to(torch.bfloat16), torch.sin(angles).to(torch.bfloat16)


def apply_rope(
    q: torch.Tensor,         # (B, H, L, D_head)
    k: torch.Tensor,         # (B, H, L, D_head)
    rope_pos: torch.Tensor,  # (B, L) int32
    cos_table: torch.Tensor, # (max_seq_len, D_head // 2)
    sin_table: torch.Tensor, # (max_seq_len, D_head // 2)
):
    # Gather per-document angles -> (B, 1, L, D_head // 2)
    cos = cos_table[rope_pos].unsqueeze(1)
    sin = sin_table[rope_pos].unsqueeze(1)

    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot
