#!/usr/bin/env python3
"""
Filter ArXiv JSONL metadata by ArXiv category code(s).

Usage:
    uv run filter.py --download --verbose
    uv run filter.py --categories physics.acc-ph hep-ex --verbose
    uv run filter.py \
        --input data/arxiv-metadata-oai-snapshot.json \
        --output out/filtered.jsonl
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

DATASET = "Cornell-University/arxiv"
DEFAULT_INPUT = Path(__file__).parent / "data" / "arxiv-metadata-oai-snapshot.json"
DEFAULT_CATEGORIES = ["physics.acc-ph"]
CHUNK_SIZE = 50_000


def download_dataset(input_path: Path, verbose: bool = False) -> None:
    if input_path.exists():
        if verbose:
            print(f"Dataset already exists at {input_path}, skipping download.")
        return

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        sys.exit(
            "Kaggle credentials not found. Create ~/.kaggle/kaggle.json with "
            '{"username": "...", "key": "..."} and chmod 600 it.\n'
            "See: https://www.kaggle.com/docs/api"
        )

    import kaggle  # type: ignore

    input_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading {DATASET} to {input_path.parent} ...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET, path=str(input_path.parent), unzip=True, quiet=not verbose
    )
    if verbose:
        print(f"Download complete: {input_path}")


def matches_any(category_string: str, wanted: set) -> bool:
    if not isinstance(category_string, str):
        return False
    return bool(set(category_string.split()) & wanted)


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    wanted: set,
    chunksize: int,
    verbose: bool = False,
) -> dict:
    total = 0
    matched = 0
    start = time.monotonic()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader = pd.read_json(input_path, lines=True, chunksize=chunksize)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk_num, chunk in enumerate(reader, start=1):
            total += len(chunk)

            if "categories" in chunk.columns:
                mask = chunk["categories"].apply(lambda c: matches_any(c, wanted))
                filtered = chunk[mask]
                matched += len(filtered)
                for _, row in filtered.iterrows():
                    out_f.write(row.to_json() + "\n")

            if verbose:
                elapsed = time.monotonic() - start
                rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"  chunk {chunk_num:>4} | "
                    f"records: {total:>9,} | "
                    f"matched: {matched:>7,} | "
                    f"rate: {rate:>9,.0f} rec/s",
                    flush=True,
                )

    return {"total": total, "matched": matched, "elapsed": time.monotonic() - start}


def resolve_output(args) -> Path:
    if args.output:
        return args.output
    slug = "_".join(sorted(args.categories)).replace(".", "-").replace("/", "-")
    return Path(__file__).parent / "data" / f"filtered_{slug}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Filter ArXiv JSONL metadata by ArXiv category code(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        metavar="CAT",
        help="ArXiv category code(s) to include",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to input JSONL file",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSONL file",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from Kaggle before filtering",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=CHUNK_SIZE,
        help="Rows per Pandas chunk for memory control",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-chunk progress and final stats",
    )
    return p


def main():
    args = build_parser().parse_args()
    wanted = set(args.categories)
    output_path = resolve_output(args)

    if args.download:
        download_dataset(args.input, verbose=args.verbose)

    if not args.input.exists():
        sys.exit(
            f"Input file not found: {args.input}\n"
            "Run with --download to fetch it from Kaggle."
        )

    if args.verbose:
        print(f"Input:      {args.input}")
        print(f"Output:     {output_path}")
        print(f"Categories: {sorted(wanted)}")
        print(f"Chunk size: {args.chunksize:,}")
        print()

    stats = filter_jsonl(
        input_path=args.input,
        output_path=output_path,
        wanted=wanted,
        chunksize=args.chunksize,
        verbose=args.verbose,
    )

    if args.verbose:
        match_rate = stats["matched"] / stats["total"] * 100 if stats["total"] else 0
        print()
        print(f"Total records processed : {stats['total']:>9,}")
        print(f"Matched records         : {stats['matched']:>9,}")
        print(f"Match rate              : {match_rate:.2f}%")
        print(f"Elapsed                 : {stats['elapsed']:.1f}s")
        print(f"Output written to       : {output_path}")


if __name__ == "__main__":
    main()
