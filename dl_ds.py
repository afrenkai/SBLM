import argparse
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

EOT_TOKEN = 50256  # GPT-2 <|endoftext|>


def write_datafile(filename: str, toks: np.ndarray):
    """
    Saves token data as a .bin file in modded-nanogpt speedrun format:
    - Header: 256 int32s (magic=20240520, version=1, token_count)
    - Payload: uint16 token IDs
    """
    assert len(toks) < 2**31, "token count too large (~2.1B max per shard)"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks) # number of uint16 tokens following header

    if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
        arr = np.asarray(toks, dtype=np.int64)
        assert arr.min() >= 0 and arr.max() < 2**16, "token ID out of uint16 range"
        toks_np = arr.astype(np.uint16)
    else:
        toks_np = toks

    print(f"writing {len(toks_np):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def main(args):
    assert args.version in ["small", "regular"], "version must be 'small' or 'regular'"

    if args.version == "small":
        ds = load_dataset(
            "nvidia/Nemotron-ClimbMix",
            data_files={"train": "climbmix_small/*.tokenized.parquet"},
            split="train",
            streaming=True,
        )
        local_dir = "climbmix_small_bins"
    else:
        ds = load_dataset(
            "nvidia/Nemotron-ClimbMix",
            split="train",
            streaming=True,
        )
        local_dir = "climbmix_regular_bins"

    data_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_dir)
    os.makedirs(data_cache_dir, exist_ok=True)

    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

    for row in ds:
        raw_toks = row["tokens"]
        if not raw_toks:
            continue

        # Prepend <|endoftext|> (50256) to mark document start
        doc_len = len(raw_toks) + 1
        tokens = np.empty((doc_len,), dtype=np.uint16)
        tokens[0] = EOT_TOKEN
        tokens[1:] = raw_toks

        # A single document could exceed shard_size in rare edge cases; loop while leftovers remain
        offset = 0
        while offset < doc_len:
            space_left = args.shard_size - token_count
            take = min(doc_len - offset, space_left)

            all_tokens_np[token_count : token_count + take] = tokens[offset : offset + take]
            token_count += take
            offset += take
            progress_bar.update(take)

            if token_count == args.shard_size:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(data_cache_dir, f"climbmix_{split}_{shard_index:06d}.bin")
                progress_bar.close()
                write_datafile(filename, all_tokens_np)

                shard_index += 1
                token_count = 0
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")

    progress_bar.close()
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_cache_dir, f"climbmix_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nemotron-ClimbMix .bin preprocessing")
    parser.add_argument("-v", "--version", type=str, default="small", help="'small' or 'regular'")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Tokens per .bin shard")
    args = parser.parse_args()
    main(args)
