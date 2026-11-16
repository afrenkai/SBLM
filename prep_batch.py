import torch
from torch.nn.attention.flex_attention import create_block_mask


@torch.no_grad()
def prepare_diffusion_batch(
    input_ids: torch.Tensor,   # (B, L) int64 on CUDA
    cu_seqlens: torch.Tensor,  # (num_docs + 1,) int32 on CUDA
    prefix_prob: float = 0.30,
):
    B, L = input_ids.shape
    device = input_ids.device

    # 1. Document IDs & per-document RoPE positions
    positions = torch.arange(B * L, device=device, dtype=torch.int32)
    doc_ids = torch.bucketize(positions, cu_seqlens[1:-1], right=True)
    rope_pos = (positions - cu_seqlens[doc_ids]).view(B, L)

    # 2. FlexAttention bidirectional document BlockMask
    doc_ids_2d = doc_ids.view(B, L)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        return doc_ids_2d[b, q_idx] == doc_ids_2d[b, kv_idx]

    block_mask = create_block_mask(
        doc_mask_mod,
        B=B,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        device=device,
        _compile=True,
    )

    # 3. Per-document logit-normal timesteps: tau_d = sigmoid(N(0, 1))
    doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1]  # (num_docs,)
    num_docs = doc_lengths.shape[0]
    tau_docs = torch.sigmoid(torch.randn(num_docs, device=device))
    tau = tau_docs[doc_ids].view(B, L)

    # 4. Vectorized prefix-inpainting mask (30% of docs with length >= 2)
    has_prefix = (torch.rand(num_docs, device=device) < prefix_prob) & (doc_lengths >= 2)
    max_prefix = torch.clamp(doc_lengths // 2, min=1)
    prefix_len_docs = (torch.rand(num_docs, device=device) * max_prefix).long() + 1
    prefix_len_docs = torch.where(has_prefix, prefix_len_docs, torch.zeros_like(prefix_len_docs))

    is_prefix = rope_pos < prefix_len_docs[doc_ids].view(B, L)
    tau = torch.where(is_prefix, torch.ones_like(tau), tau)
    labels = torch.where(is_prefix, torch.full_like(input_ids, -100), input_ids)

    return labels, tau, rope_pos, block_mask
