"""Enrich a JSONL dataset with the full OpenAlex Work object for each paper

Input: ArXiv metadata JSONL
Output: Input records with an added `openalex` key containing the full OpenAlex response

Authentication:
  OpenAlex requires an API key
  Set OPENALEX_API_KEY in a .env file or the environment, or pass --api-key
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_S = 1.5
DEFAULT_MAX_BACKOFF_S = 60.0
TITLE_MATCH_THRESHOLD = 0.7


def _norm(s: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _sanitize_query(s: str) -> str:
    s = re.sub(r"\$[^$]*\$", " ", s)
    return " ".join(s.split())


def _best_title_match(title: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    title_n = _norm(title)
    best = None
    best_score = 0.0
    for w in candidates:
        cand = w.get("display_name") or ""
        score = SequenceMatcher(None, title_n, _norm(cand)).ratio()
        if score > best_score:
            best = w
            best_score = score
    return best if best_score >= TITLE_MATCH_THRESHOLD else None


def _parse_year(record: dict[str, Any]) -> int | None:
    versions = record.get("versions")
    if isinstance(versions, list) and versions:
        v0 = versions[0]
        if isinstance(v0, dict):
            created = v0.get("created")
            if isinstance(created, str) and created.strip():
                try:
                    return parsedate_to_datetime(created).year
                except Exception:
                    pass
    return None


def _get_json(
    session: requests.Session,
    path: str,
    *,
    params: dict[str, str],
    api_key: str | None,
    timeout_s: float,
    max_retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> dict[str, Any]:
    url = f"{OPENALEX_BASE}{path}"
    if api_key:
        params = {**params, "api_key": api_key}
    attempt = 0
    while True:
        attempt += 1
        resp = session.get(url, params=params, timeout=timeout_s)
        if resp.status_code not in (429, 500):
            resp.raise_for_status()
            return resp.json()

        if attempt > max_retries:
            if resp.status_code == 429:
                raise SystemExit(
                    f"OpenAlex daily credit limit reached after {max_retries} retries"
                )
            resp.raise_for_status()

        retry_after = resp.headers.get("Retry-After")
        sleep_s = None
        if retry_after:
            try:
                sleep_s = float(retry_after)
            except Exception:
                sleep_s = None

        if sleep_s is None:
            sleep_s = min(backoff_s * (2 ** (attempt - 1)), max_backoff_s)
        else:
            sleep_s = min(sleep_s, max_backoff_s)

        print(
            f"OpenAlex {resp.status_code}. Sleeping {sleep_s:.1f}s then retrying (attempt {attempt}/{max_retries})",
            flush=True,
        )
        time.sleep(sleep_s)


def _work_by_doi(
    session: requests.Session,
    doi: str,
    *,
    timeout_s: float,
    api_key: str | None,
    max_retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> dict[str, Any] | None:
    doi = doi.strip().lower()
    if doi.startswith("http"):
        doi = doi.split("doi.org/", 1)[-1]
    if not doi:
        return None

    params = {"filter": f"doi:{doi}", "per-page": "1"}
    try:
        payload = _get_json(
            session,
            "/works",
            params=params,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            max_backoff_s=max_backoff_s,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            return None
        raise
    results = payload.get("results") or []
    return results[0] if results else None


def _work_by_title_year(
    session: requests.Session,
    title: str,
    year: int | None,
    *,
    timeout_s: float,
    api_key: str | None,
    per_page: int,
    max_retries: int,
    backoff_s: float,
    max_backoff_s: float,
) -> dict[str, Any] | None:
    q = _sanitize_query(title or "")
    if not q:
        return None

    params: dict[str, str] = {"search": q, "per-page": str(per_page)}
    if year is not None:
        params["filter"] = f"publication_year:{year}"

    payload = _get_json(
        session,
        "/works",
        params=params,
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_s=backoff_s,
        max_backoff_s=max_backoff_s,
    )
    results = payload.get("results") or []
    if not results:
        return None
    return _best_title_match(q, results)


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
    api_key: str | None
    timeout_s: float
    sleep_s: float
    per_page: int
    overwrite: bool
    max_retries: int
    backoff_s: float
    max_backoff_s: float


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Input JSONL")
    p.add_argument("--output", type=Path, required=True, help="Output JSONL")
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAlex API key (overrides OPENALEX_API_KEY env var)",
    )
    p.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout seconds")
    p.add_argument("--sleep-s", type=float, default=0.25, help="Sleep between requests")
    p.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries for 429 rate limits (default: {DEFAULT_MAX_RETRIES})",
    )
    p.add_argument(
        "--backoff-s",
        type=float,
        default=DEFAULT_BACKOFF_S,
        help=f"Initial exponential backoff seconds for 429 (default: {DEFAULT_BACKOFF_S})",
    )
    p.add_argument(
        "--max-backoff-s",
        type=float,
        default=DEFAULT_MAX_BACKOFF_S,
        help=f"Max backoff seconds for 429 (default: {DEFAULT_MAX_BACKOFF_S})",
    )
    p.add_argument(
        "--per-page",
        type=int,
        default=5,
        help="Candidate works to retrieve for title search (default: 5)",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite output instead of resume"
    )
    return p


def main() -> None:
    ns = build_parser().parse_args()
    api_key = ns.api_key or os.environ.get("OPENALEX_API_KEY") or None
    if not api_key:
        print("Warning: No OPENALEX_API_KEY Set", flush=True)
    args = Args(
        input=ns.input,
        output=ns.output,
        api_key=api_key,
        timeout_s=ns.timeout_s,
        sleep_s=ns.sleep_s,
        per_page=ns.per_page,
        overwrite=ns.overwrite,
        max_retries=ns.max_retries,
        backoff_s=ns.backoff_s,
        max_backoff_s=ns.max_backoff_s,
    )

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    checkpoint = set() if args.overwrite else _load_checkpoint(args.output)

    processed = 0
    skipped = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "accelerator-physics-arxiv/0.1 (OpenAlex)"})

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

            doi = record.get("doi")
            title = record.get("title") or ""
            year = _parse_year(record)

            work = None
            if isinstance(doi, str) and doi.strip():
                work = _work_by_doi(
                    session,
                    doi,
                    timeout_s=args.timeout_s,
                    api_key=args.api_key,
                    max_retries=args.max_retries,
                    backoff_s=args.backoff_s,
                    max_backoff_s=args.max_backoff_s,
                )
            if work is None:
                work = _work_by_title_year(
                    session,
                    title,
                    year,
                    timeout_s=args.timeout_s,
                    api_key=args.api_key,
                    per_page=args.per_page,
                    max_retries=args.max_retries,
                    backoff_s=args.backoff_s,
                    max_backoff_s=args.max_backoff_s,
                )

            enriched = dict(record)
            enriched["openalex"] = work
            f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            f_out.flush()

            checkpoint.add(rec_id)
            processed += 1

            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

            if processed % 100 == 0:
                print(f"Processed={processed:,} Skipped={skipped:,}", flush=True)

    print(f"Done: Processed={processed:,} Skipped={skipped:,} Output={args.output}", flush=True)


if __name__ == "__main__":
    main()
