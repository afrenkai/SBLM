import torch
import numpy as np


def load_bin_tokens(path: str) -> np.memmap:
    header = np.fromfile(path, count=256, dtype=np.int32)
    assert header[0] == 20240520 and header[1] == 1, f"Bad header in {path}"
    num_tokens = int(header[2])
    return np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(num_tokens,))

def get_batch_with_cuseqlens(memmap_tokens: np.memmap, B: int, L: int, device: torch.device):
    ix = torch.randint(0, len(memmap_tokens) - B * L - 1, (1,)).item()
    flat = torch.from_numpy(memmap_tokens[ix : ix + B * L].astype(np.int64)).to(device)

    # Split at every <|endoftext|> (50256) AND every row boundary (0, L, 2L, ..., B*L)
    eot_pos = torch.nonzero(flat == 50256, as_tuple=False).squeeze(-1).to(torch.int32)
    row_bounds = torch.arange(0, B * L + 1, L, device=device, dtype=torch.int32)
    cu_seqlens = torch.unique(torch.cat([row_bounds, eot_pos]), sorted=True)

    return flat.view(B, L), cu_seqlens
