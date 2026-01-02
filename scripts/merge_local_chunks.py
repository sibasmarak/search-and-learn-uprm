import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import datasets


RANGE_RE = re.compile(r"range-(\d+)-(\d+)$")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in {path} line {line_no}: {e}") from e


def find_chunks(data_completions_dir: Path, filename: str) -> List[Tuple[int, int, Path]]:
    """
    Returns list of (start, end, jsonl_path) sorted by start.
    Expects folder names like range-0-25 containing dvts_completions.jsonl
    """
    chunks: List[Tuple[int, int, Path]] = []
    for sub in data_completions_dir.iterdir():
        if not sub.is_dir():
            continue
        m = RANGE_RE.match(sub.name)
        if not m:
            continue
        start, end = int(m.group(1)), int(m.group(2))
        jsonl_path = sub / filename
        if jsonl_path.exists():
            chunks.append((start, end, jsonl_path))
        else:
            raise FileNotFoundError(f"Missing {filename} in {sub}")
    chunks.sort(key=lambda x: x[0])
    return chunks


def load_chunks_as_dataset(chunks: List[Tuple[int, int, Path]]) -> datasets.Dataset:
    """
    Loads all JSONL rows into a single Dataset.
    Uses streaming generator -> Dataset.from_list in blocks to avoid huge RAM spikes.
    """
    # Build rows in blocks (you can tune block_size if needed)
    block_size = 50_000
    rows: List[Dict[str, Any]] = []
    dsets: List[datasets.Dataset] = []

    for chunk_id, (start, end, jsonl_path) in enumerate(chunks):
        for ex in iter_jsonl(jsonl_path):
            rows.append(ex)

            if len(rows) >= block_size:
                dsets.append(datasets.Dataset.from_list(rows))
                rows = []

    if rows:
        dsets.append(datasets.Dataset.from_list(rows))

    if not dsets:
        return datasets.Dataset.from_list([])

    if len(dsets) == 1:
        return dsets[0]

    return datasets.concatenate_datasets(dsets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_completions_dir",
        type=Path,
        required=True,
        help="Directory containing range-*-* subfolders",
    )
    ap.add_argument(
        "--filename",
        type=str,
        default="dvts_completions.jsonl",
        help="Chunk JSONL filename inside each range folder",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use as config_name",
    )
    ap.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HF dataset repo id, e.g. sibasmarakp/Qwen2.5-1.5B-Instruct-dvts-completions",
    )
    ap.add_argument("--split", type=str, default="train", help="Dataset split to push")
    ap.add_argument("--private", action="store_true", help="Push as private dataset")
    ap.add_argument(
        "--expected_num_samples",
        type=int,
        default=500,
        help="Expected number of samples in the dataset (used for sanity check)",
    )
    ap.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4_MATH-500",
        help="Dataset name to use for the config",
    )
    ap.add_argument(
        "--agg_strategy",
        type=str,
        default="last",
        help="Aggregation strategy to use (options: last, min, prod)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature to use for the dataset",
    )
    args = ap.parse_args()

    chunks = find_chunks(args.data_completions_dir, args.filename)
    if not chunks:
        raise SystemExit(f"No chunks found under {args.data_completions_dir} matching range-*-*/{args.filename}")

    print(f"Found {len(chunks)} chunks, merging...")
    ds = load_chunks_as_dataset(chunks)

    if "problem" in ds.column_names and len(ds.unique("problem")) != len(ds):
        raise ValueError("Found duplicate problems")
    if len(ds) != args.expected_num_samples:
        raise ValueError(f"Expected {args.expected_num_samples} samples, got {len(ds)}")

    print(f"Merged rows: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    # Push: use config_name = seed if provided, else 'default'
    cols = [col for col in ds.column_names if "pred_maj@" in col]
    n = max(int(col.split("@")[-1]) for col in cols)
    config_name = f"{args.dataset_name}--T-{args.temperature}--top_p-1.0--n-{n}--seed-{args.seed}--agg_strategy-{args.agg_strategy}"
    print(f"Pushing to: {config_name}")

    url = ds.push_to_hub(
        repo_id=args.repo_id,
        config_name=config_name,
        split=args.split,
        private=args.private,
    )
    print(f"Pushed to: {url}")


if __name__ == "__main__":
    main()
