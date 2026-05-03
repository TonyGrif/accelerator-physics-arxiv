"""Annotate an arXiv accelerator-physics JSONL with explicit AI/ML method mentions"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

Matches = dict[str, list[str]]


def _compile(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


METHOD_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "ML": _compile(
        [
            r"\bmachine learning\b",
            r"\bdeep learning\b",
            r"\bneural networks?\b",
            r"\breinforcement learning\b",
            r"\bgaussian process(?: regression)?\b",
            r"\bsupport vector machine\b",
            r"\bSVM\b",
            r"\brandom forest\b",
            r"\bxgboost\b",
            r"\bgradient boosting\b",
            r"\bautoencoders?\b",
            r"\bconvolutional neural network\b",
            r"\bCNN\b",
            r"\brecurrent neural network\b",
            r"\bRNN\b",
            r"\blong short-term memory\b",
            r"\bLSTM\b",
            r"\bgenerative adversarial network\b",
            r"\bGAN\b",
            r"\bvariational autoencoder\b",
            r"\bVAE\b",
            r"\bgraph neural network\b",
            r"\bGNN\b",
            r"\btransformer (network|model|architecture)\b",
            r"\battention mechanism\b",
        ]
    ),
    "AI_OPT": _compile(
        [
            r"\bbayesian optimization\b",
            r"\bgenetic algorithm(?:s)?\b",
            r"\bevolutionary algorithm(?:s)?\b",
            r"\bevolutionary strateg(?:y|ies)\b",
            r"\bparticle swarm optimization\b",
        ]
    ),
    "SURR": _compile(
        [
            r"\bsurrogate model(?:s)?\b",
            r"\bsurrogate-based\b",
            r"\bPINN\b",
        ]
    ),
    "DTWIN": _compile([r"\bdigital twin(?:s)?\b"]),
    "DS_TOOLS": _compile(
        [
            r"\bPyTorch\b",
            r"\bTensorFlow\b",
            r"\bKeras\b",
            r"\bscikit-learn\b",
            r"\bsklearn\b",
            r"\bONNX\b",
            r"\bHugging Face\b",
        ]
    ),
    "DS_STAT": _compile(
        [
            r"\bprincipal component analysis\b",
            r"\bdimensionality reduction\b",
            r"\bk-means\b",
            r"\bDBSCAN\b",
            r"\bfeature (extraction|engineering|selection|importance)\b",
            r"\bcross[- ]validation\b",
            r"\bhyperparameter\b",
            r"\bprecision[- ]recall\b",
            r"\bROC curve\b",
        ]
    ),
}


def _text_for_matching(record: dict) -> str:
    title = record.get("title") or ""
    abstract = record.get("abstract") or ""
    return f"{title}\n{abstract}"


def _match_patterns(
    text: str, pattern_map: dict[str, list[re.Pattern[str]]]
) -> Matches:
    hits: Matches = {}
    for code, patterns in pattern_map.items():
        matched: list[str] = []
        for pat in patterns:
            if pat.search(text):
                matched.append(pat.pattern)
        if matched:
            hits[code] = matched
    return hits


def _load_checkpoint(output_path: Path) -> set[str]:
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                seen.add(str(obj.get("id")))
            except Exception:
                continue
    return seen


@dataclass(frozen=True)
class Args:
    input: Path
    output: Path
    overwrite: bool
    include_matches: bool
    progress_every: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Input JSONL")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL")
    p.add_argument(
        "--include-matches",
        action="store_true",
        help="Include matched pattern strings in output for auditing",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output instead of resume",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N records (default: 100)",
    )
    return p


def main() -> None:
    ns = build_parser().parse_args()
    args = Args(
        input=ns.input,
        output=ns.output,
        overwrite=ns.overwrite,
        include_matches=ns.include_matches,
        progress_every=ns.progress_every,
    )

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    checkpoint: set[str] = set() if args.overwrite else _load_checkpoint(args.output)

    processed = 0
    skipped = 0

    with (
        args.input.open(encoding="utf-8") as f_in,
        args.output.open(mode, encoding="utf-8") as f_out,
    ):
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            rec_id = str(record.get("id"))
            if rec_id in checkpoint:
                skipped += 1
                continue

            text = _text_for_matching(record)
            method_matches = _match_patterns(text, METHOD_PATTERNS)
            methods = sorted(method_matches.keys())

            enriched = dict(record)
            enriched["methods"] = methods
            enriched["explicit_ds_any"] = 1 if methods else 0
            if args.include_matches:
                enriched["explicit_ai_matches"] = method_matches

            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            checkpoint.add(rec_id)
            processed += 1

            if processed % args.progress_every == 0:
                print(f"processed={processed:,} skipped={skipped:,}", flush=True)

    print(
        f"done processed={processed:,} skipped={skipped:,} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
