import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask


@dataclass
class DDTConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    enc_layers: int = 10
    dec_layers: int = 3
    mlp_hidden: int = 2048   # Exact 8/3 * 768 for SwiGLU, multiple of 256
    max_seq_len: int = 2048
    rope_theta: float = 10000.0


def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(torch.bfloat16), torch.sin(angles).to(torch.bfloat16)


def apply_rope(
    q: torch.Tensor,          # (B, H, L, D_head)
    k: torch.Tensor,          # (B, H, L, D_head)
    rope_pos: torch.Tensor,   # (B, L) int32
    cos_table: torch.Tensor,  # (max_seq_len, D_head // 2)
    sin_table: torch.Tensor,  # (max_seq_len, D_head // 2)
):
    cos = cos_table[rope_pos].unsqueeze(1)  # (B, 1, L, D_head // 2)
    sin = sin_table[rope_pos].unsqueeze(1)
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


class TimestepEmbedder(nn.Module):
    """Maps continuous tau in [0, 1] to (B, L, d_model) + shared 6-way AdaLN modulation."""
    def __init__(self, d_model: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.shared_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init shared AdaLN projection so blocks start near identity
        nn.init.zeros_(self.shared_adaln[1].weight)
        nn.init.zeros_(self.shared_adaln[1].bias)

    def forward(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tau: (B, L) -> sinusoidal embedding (B, L, freq_dim)
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
        )
        args = tau.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(emb)
        shared_mod = self.shared_adaln(t_emb)  # (B, L, 6 * d_model)
        return t_emb, shared_mod


class DiTBlock(nn.Module):
    def __init__(self, cfg: DDTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0

        self.wqkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # SwiGLU FFN
        self.w13 = nn.Linear(cfg.d_model, 2 * cfg.mlp_hidden, bias=False)
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

        # Lightweight per-layer AdaLN offset (replaces bulky per-layer d x 6d linear)
        self.adaln_bias = nn.Parameter(torch.zeros(1, 1, 6 * cfg.d_model))

        # Zero-init residual output projections for stable deep training
        nn.init.zeros_(self.wo.weight)
        nn.init.zeros_(self.w2.weight)

    def forward(
        self,
        x: torch.Tensor,
        shared_mod: torch.Tensor,
        rope_pos: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shared_mod + self.adaln_bias
        ).chunk(6, dim=-1)

        # 1. Pre-RMSNorm + AdaLN modulation + Self-Attention
        h = F.rms_norm(x, (D,)) * (1.0 + scale_msa) + shift_msa
        q, k, v = self.wqkv(h).chunk(3, dim=-1)

        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Fused QK-Norm -> 1D per-doc RoPE -> FlexAttention
        q = F.rms_norm(q, (self.d_head,))
        k = F.rms_norm(k, (self.d_head,))
        q, k = apply_rope(q, k, rope_pos, cos_table, sin_table)

        attn_out = flex_attention(q, k, v, block_mask=block_mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + gate_msa * self.wo(attn_out)

        # 2. Pre-RMSNorm + AdaLN modulation + SwiGLU MLP
        h = F.rms_norm(x, (D,)) * (1.0 + scale_mlp) + shift_mlp
        gate, up = self.w13(h).chunk(2, dim=-1)
        x = x + gate_mlp * self.w2(F.silu(gate) * up)

        return x


class BrownianDDT(nn.Module):
    def __init__(self, cfg: DDTConfig):
        super().__init__()
        self.cfg = cfg

        # Unconstrained embedding table (regularized toward N(0, I) by SIGReg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0)

        self.time_embed = TimestepEmbedder(cfg.d_model)
        self.dec_cond_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 6 * cfg.d_model, bias=False),
        )
        nn.init.zeros_(self.dec_cond_proj[2].weight)

        self.encoder = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.enc_layers)])
        self.decoder = nn.ModuleList([DiTBlock(cfg) for _ in range(cfg.dec_layers)])

        # Fused Linear + CrossEntropy head for auxiliary encoder supervision (saves HBM)
        self.aux_enc_ce = nn.LinearCrossEntropyLoss(
            cfg.d_model, cfg.vocab_size, bias=False, ignore_index=-100
        )

        # Main decoder head (returns logits so we can compute both CE and softmax @ E)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

        cos_table, sin_table = precompute_rope_freqs(
            cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def encode(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        rope_pos: torch.Tensor,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        _, shared_mod = self.time_embed(tau)
        z = x_t
        for block in self.encoder:
            z = block(z, shared_mod, rope_pos, self.cos_table, self.sin_table, block_mask)
        return F.rms_norm(z, (self.cfg.d_model,))

    def decode(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        z_t: torch.Tensor,
        rope_pos: torch.Tensor,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        _, shared_mod = self.time_embed(tau)
        dec_mod = shared_mod + self.dec_cond_proj(z_t)
        h = x_t
        for block in self.decoder:
            h = block(h, dec_mod, rope_pos, self.cos_table, self.sin_table, block_mask)
        h = F.rms_norm(h, (self.cfg.d_model,))
        return self.lm_head(h)

    def forward(
        self,
        x_t: torch.Tensor,
        tau: torch.Tensor,
        rope_pos: torch.Tensor,
        block_mask: BlockMask | None = None,
        labels: torch.Tensor | None = None,
    ):
        z_t = self.encode(x_t, tau, rope_pos, block_mask)
        logits = self.decode(x_t, tau, z_t, rope_pos, block_mask)

        aux_loss = None
        if labels is not None:
            aux_loss = self.aux_enc_ce(
                z_t.reshape(-1, self.cfg.d_model),
                labels.reshape(-1),
            )
        return logits, z_t, aux_loss

    def get_param_groups(self, dion_lr: float = 1.5e-3, adam_lr: float = 3e-4):
        """Splits parameters for Dion (2D hidden matrices) vs AdamW (vocab tables & 1D params)."""
        dion_params, adam_decay, adam_no_decay = [], [], []
        vocab_ids = {
            id(self.tok_emb.weight),
            id(self.lm_head.weight),
            id(self.aux_enc_ce.weight),
        }
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in vocab_ids:
                adam_decay.append(p)
            elif p.ndim == 2:
                dion_params.append(p)
            else:
                adam_no_decay.append(p)

        return {
            "dion": [{"params": dion_params, "lr": dion_lr, "weight_decay": 0.05}],
            "adamw": [
                {"params": adam_decay, "lr": adam_lr, "weight_decay": 0.1},
                {"params": adam_no_decay, "lr": adam_lr, "weight_decay": 0.0},
            ],
        }
