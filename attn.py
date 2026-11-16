import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention
from einops import rearrange
from torch import Tensor


class FusedLanguageAttention(MiniMaxH3Attention):
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        freqs_cis: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        return super().forward(
            hidden_states, rotary_emb=freqs_cis, attention_mask=attention_mask
        )


@torch.no_grad()
def fuse_language_attention(
    attention: Attention, width: int, heads: int
) -> FusedLanguageAttention:
    fused = FusedLanguageAttention(width, heads, width // heads)
    fused.to_q.weight.copy_(
        rearrange(
            attention.to_q.weight,
            "(head pair component) input -> (head component pair) input",
            head=heads,
            component=2,
        )
    )
    fused.to_k.weight.copy_(
        rearrange(
            attention.to_k.weight,
            "(head pair component) input -> (head component pair) input",
            head=heads,
            component=2,
        )
    )
    fused.to_v.weight.copy_(attention.to_v.weight)
    fused.norm_q.weight.copy_(
        rearrange(
            attention.norm_q.weight, "(pair component) -> (component pair)", component=2
        )
    )
    fused.norm_k.weight.copy_(
        rearrange(
            attention.norm_k.weight, "(pair component) -> (component pair)", component=2
        )
    )
    fused.to_out[0].weight.copy_(attention.to_out[0].weight)
    fused.fuse_projections()
    del fused.to_q
    del fused.to_k
    del fused.to_v
 
