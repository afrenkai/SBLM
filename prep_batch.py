import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def build_document_mask(doc_ids: torch.Tensor, is_prefix: torch.Tensor) -> BlockMask:
    batch_size, sequence_length = doc_ids.shape

    def mask_mod(
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        bounded_query = query_index.clamp(max=sequence_length - 1)
        bounded_key = key_index.clamp(max=sequence_length - 1)
        same_document = doc_ids[batch_index, bounded_query] == doc_ids[batch_index, bounded_key]
        prefix_query = is_prefix[batch_index, bounded_query]
        prefix_key = is_prefix[batch_index, bounded_key]
        return (
            (query_index < sequence_length)
            & (key_index < sequence_length)
            & same_document
            & (~prefix_query | (prefix_key & (key_index <= query_index)))
        )

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=doc_ids.device,
        _compile=True,
    )


@torch.no_grad()
def prepare_diffusion_batch(
    input_ids: torch.Tensor,  # (B, L) int64 on CUDA
    cu_seqlens: torch.Tensor,  # (num_docs + 1,) int32 on CUDA
    prefix_prob: float = 0.30,
    logit_norm_mean: float = 0.0,
):
    B, L = input_ids.shape
    device = input_ids.device

    positions = torch.arange(B * L, device=device, dtype=torch.int32)
    doc_ids = torch.bucketize(positions, cu_seqlens[1:-1], right=True)
    rope_pos = (positions - cu_seqlens[doc_ids]).view(B, L)

    doc_lengths = cu_seqlens[1:] - cu_seqlens[:-1]  # (num_docs,)
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
    block_mask = build_document_mask(doc_ids.view(B, L), is_prefix)

    return labels, tau, rope_pos, block_mask
