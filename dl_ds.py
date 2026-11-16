import argparse
import gc
import os
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "nvidia/Nemotron-ClimbMix"
EOT_TOKEN = 50256  # GPT-2 <|endoftext|>
MAGIC = 20240520
VERSION = 1


def is_valid_bin(path: str) -> bool:
    """Checks if an output .bin file exists, has a valid header, and isn't truncated."""
    if not os.path.isfile(path) or os.path.getsize(path) < 256 * 4:
        return False
    header = np.fromfile(path, count=256, dtype=np.int32)
    if header[0] != MAGIC or header[1] != VERSION:
        return False
    expected_bytes = 256 * 4 + int(header[2]) * 2
    return os.path.getsize(path) == expected_bytes


def write_datafile_atomic(filename: str, toks_np: np.ndarray):
    assert len(toks_np) < 2**31, f"Shard too large ({len(toks_np):,} tokens)"
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(toks_np)

    tmp_path = filename + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())
    os.replace(tmp_path, filename)
    print(f"wrote {len(toks_np):,} tokens -> {filename}", flush=True)


def parquet_to_uint16_with_eot(parquet_path: str) -> np.ndarray:
    """Converts a tokenized Parquet shard directly to flat uint16 with <|endoftext|> prepended."""
    pf = pq.ParquetFile(parquet_path)
    chunks = []

    for batch in pf.iter_batches(batch_size=8192, columns=["tokens"]):
        col = batch.column("tokens")
        offsets = col.offsets.to_numpy(zero_copy_only=False)
        values = col.values.to_numpy(zero_copy_only=False)

        lengths = offsets[1:] - offsets[:-1]
        valid = lengths > 0
        if not np.any(valid):
            continue

        valid_lengths = lengths[valid]
        flat_vals = values[offsets[0] : offsets[-1]]

        total = int(valid_lengths.sum()) + len(valid_lengths)
        eot_indices = np.cumsum(valid_lengths + 1) - (valid_lengths + 1)

        mask = np.ones(total, dtype=bool)
        mask[eot_indices] = False

        out = np.empty(total, dtype=np.uint16)
        out[eot_indices] = EOT_TOKEN
        out[mask] = flat_vals
        chunks.append(out)

    del pf
    if not chunks:
        return np.empty((0,), dtype=np.uint16)
    return np.concatenate(chunks, axis=0)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main(args):
    assert args.version in ["small", "regular"], "version must be 'small' or 'regular'"
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"climbmix_{args.version}_bins"
    )
    tmp_dir = os.path.join(out_dir, "_tmp_parquet")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")

    if args.version == "small":
        parquet_files = sorted(
            [f for f in all_files if f.startswith("climbmix_small/") and f.endswith(".parquet")],
            key=natural_key,
        )
    else:
        parquet_files = sorted(
            [f for f in all_files if not f.startswith("climbmix_small/") and f.endswith(".parquet")],
            key=natural_key,
        )

    if args.max_shards is not None:
        parquet_files = parquet_files[: args.max_shards]

    print(f"Found {len(parquet_files)} Parquet shards for version='{args.version}'", flush=True)

    for idx, repo_file in enumerate(parquet_files):
        split = "val" if idx == 0 else "train"
        out_path = os.path.join(out_dir, f"climbmix_{split}_{idx:06d}.bin")

        if is_valid_bin(out_path):
            print(f"[{idx+1}/{len(parquet_files)}] skipping existing {out_path}", flush=True)
            continue

        print(f"[{idx+1}/{len(parquet_files)}] downloading {repo_file}...", flush=True)
        local_pq = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file,
            repo_type="dataset",
            local_dir=tmp_dir,
        )

        toks_np = parquet_to_uint16_with_eot(local_pq)
        write_datafile_atomic(out_path, toks_np)

        # Clean up temp Parquet and C++ Arrow memory immediately
        del toks_np
        if os.path.exists(local_pq):
            os.remove(local_pq)
        pa.default_memory_pool().release_unused()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumable Nemotron-ClimbMix .bin builder")
    parser.add_argument("-v", "--version", type=str, default="small", choices=["small", "regular"])
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_shards", type=int, default=None, help="Cap number of shards (e.g. 60 for ~35B tokens)")
    args = parser.parse_args()
    main(args)
