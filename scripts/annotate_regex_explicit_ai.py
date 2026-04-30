"""Annotate arXiv accelerator physics JSONL with AI/ML mentions

Usage:
  python3 scripts/annotate_regex_explicit_ai.py \
    --input data/filtered_physics-acc-ph.jsonl \
    --output data/annotated_physics-acc-ph_regex.jsonl \
    --include-matches
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

DEFAULT_INPUT = Path(__file__).parent.parent / "data" / "filtered_physics-acc-ph.jsonl"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "data" / "annotated_physics-acc-ph_regex.jsonl"
)


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
            r"\bGP\b",
            r"\bsupport vector machine\b",
            r"\bSVM\b",
            r"\brandom forest\b",
            r"\bxgboost\b",
            r"\bgradient boosting\b",
            r"\bautoencoders?\b",
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
        ]
    ),
    "DTWIN": _compile([r"\bdigital twin(?:s)?\b"]),
}

CONTEXT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "DIAG": _compile(
        [
            r"\bdiagnostic(?:s)?\b",
            r"\bbeam diagnostics\b",
            r"\bbpm\b",
            r"\bprofile\b",
            r"\bimage(?:s|ing)?\b",
            r"\btomograph(?:y|ic)\b",
            r"\breconstruction\b",
            r"\binversion\b",
        ]
    ),
    "CTRL": _compile(
        [
            r"\bcontrol\b",
            r"\bcontroller\b",
            r"\bfeedback\b",
            r"\bclosed[- ]loop\b",
            r"\bmodel predictive control\b",
        ]
    ),
    "OPT": _compile(
        [
            r"\btuning\b",
            r"\boptimization\b",
            r"\bonline optimization\b",
            r"\bparameter optimization\b",
        ]
    ),
    "SIM": _compile(
        [
            r"\bsimulation\b",
            r"\bemulator\b",
            r"\breduced[- ]order\b",
            r"\bfast (model|modeling)\b",
            r"\bfast modeling\b",
        ]
    ),
    "DATA": _compile(
        [
            r"\barchived data\b",
            r"\barchive data\b",
            r"\bhistorical data\b",
            r"\blog data\b",
            r"\bdata analysis\b",
            r"\bdata mining\b",
            r"\blarge (quantities|amounts) of data\b",
        ]
    ),
    "ANOM": _compile(
        [
            r"\banomaly detection\b",
            r"\bfault detection\b",
            r"\boutlier\b",
            r"\bclassification\b",
        ]
    ),
    "DESIGN": _compile(
        [
            r"\blattice design\b",
            r"\binjector\b",
            r"\bphotoinjector\b",
            r"\bgunn?\b",
            r"\bmagnet design\b",
            r"\brf systems?\b",
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate JSONL with explicit AI/ML mentions via regex"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL")
    p.add_argument(
        "--include-matches",
        action="store_true",
        help="Include explicit_ai_matches (pattern strings) for auditing",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output instead of append/resume",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N records",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.overwrite else "a"
    checkpoint: set[str] = set() if args.overwrite else _load_checkpoint(args.output)

    processed = 0
    skipped = 0
    total_read = 0

    with (
        args.input.open(encoding="utf-8") as f_in,
        args.output.open(mode, encoding="utf-8") as f_out,
    ):
        for line in f_in:
            if not line.strip():
                continue
            total_read += 1
            record = json.loads(line)
            rec_id = str(record.get("id"))
            if rec_id in checkpoint:
                skipped += 1
                continue

            text = _text_for_matching(record)
            method_matches = _match_patterns(text, METHOD_PATTERNS)
            methods = sorted(method_matches.keys())
            has_methods = bool(methods)

            context_matches: Matches = {}
            contexts: list[str] = []
            if has_methods:
                context_matches = _match_patterns(text, CONTEXT_PATTERNS)
                contexts = sorted(context_matches.keys())

            matches: Matches = {}
            matches.update(method_matches)
            matches.update(context_matches)

            enriched = dict(record)
            enriched["methods"] = methods
            enriched["contexts"] = contexts
            enriched["explicit_ai_any"] = 1 if has_methods else 0
            if args.include_matches:
                enriched["explicit_ai_matches"] = matches

            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            processed += 1
            checkpoint.add(rec_id)

            if args.progress_every > 0 and processed % args.progress_every == 0:
                print(
                    f"Processed={processed:,} Skipped={skipped:,} Read={total_read:,}",
                    flush=True,
                )

    print(
        f"Processed={processed:,} Skipped={skipped:,} Read={total_read:,} Output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
