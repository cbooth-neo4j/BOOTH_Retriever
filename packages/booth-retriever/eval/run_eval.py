"""Manual evaluation harness for BOOTH.

Run ``python eval/run_eval.py --help`` for usage. Not part of pytest; this
script calls a real Neo4j and a real LLM.

Exit codes:
    0  - ran to completion
    1  - usage error or IO error
    2  - all questions missed the cache (probably indicates a setup issue)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional dep, user guidance
    print(
        "PyYAML is required for the eval harness. "
        "Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


@dataclass
class QuestionResult:
    text: str
    latency_ms: float
    cache_hit: bool
    answer: str
    expected_keywords: list[str]
    expected_to_hit: bool
    keyword_match: bool
    query_id: str | None
    declined: bool
    error: str | None


@dataclass
class RunSummary:
    total: int
    cache_hits: int
    keyword_matches_on_hits: int
    cache_hit_rate: float
    keyword_precision_on_hits: float
    p50_latency_ms: float
    p95_latency_ms: float


def _env_or_exit(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"Missing environment variable: {name}", file=sys.stderr)
        raise SystemExit(1)
    return v


def _build_retriever():
    """Build a real BOOTHRetriever from env. Swap embedders/LLMs as needed."""
    from neo4j import GraphDatabase
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings

    from booth_retriever import BOOTHRetriever

    driver = GraphDatabase.driver(
        _env_or_exit("NEO4J_URI"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), _env_or_exit("NEO4J_PASSWORD")),
    )
    embedder = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return BOOTHRetriever(driver=driver, embedder=embedder), driver


def _load_testset(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"Testset not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    if not questions:
        print(f"Testset has no questions: {path}", file=sys.stderr)
        raise SystemExit(1)
    return questions


def _keyword_match(answer: str, keywords: list[str]) -> bool:
    """Simple string-contains match; swap for an LLM judge in real evals."""
    if not keywords:
        return True
    haystack = (answer or "").lower()
    return all(kw.lower() in haystack for kw in keywords)


def run(testset_path: Path, out_path: Path | None) -> int:
    questions = _load_testset(testset_path)
    retriever, driver = _build_retriever()

    results: list[QuestionResult] = []
    try:
        for q in questions:
            text = q["text"]
            expected_keywords = list(q.get("expected_keywords", []))
            expected_to_hit = bool(q.get("expected_to_hit", False))

            start = time.perf_counter()
            try:
                resp = retriever.query(text)
                err = None
            except Exception as exc:  # noqa: BLE001 - eval should not crash mid-run
                resp = None
                err = f"{type(exc).__name__}: {exc}"
            elapsed_ms = (time.perf_counter() - start) * 1000

            if resp is None:
                results.append(
                    QuestionResult(
                        text=text,
                        latency_ms=elapsed_ms,
                        cache_hit=False,
                        answer="",
                        expected_keywords=expected_keywords,
                        expected_to_hit=expected_to_hit,
                        keyword_match=False,
                        query_id=None,
                        declined=False,
                        error=err,
                    )
                )
                continue

            results.append(
                QuestionResult(
                    text=text,
                    latency_ms=elapsed_ms,
                    cache_hit=bool(resp.similar_match and resp.success),
                    answer=resp.answer,
                    expected_keywords=expected_keywords,
                    expected_to_hit=expected_to_hit,
                    keyword_match=_keyword_match(resp.answer, expected_keywords),
                    query_id=resp.query_id,
                    declined=resp.declined,
                    error=None,
                )
            )
    finally:
        driver.close()

    summary = _summarise(results)
    _print_table(results, summary)

    if out_path is not None:
        payload = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {out_path}")

    return 0 if summary.cache_hits else 2


def _summarise(results: list[QuestionResult]) -> RunSummary:
    total = len(results)
    hits = [r for r in results if r.cache_hit]
    keyword_hits = [r for r in hits if r.keyword_match]
    latencies = sorted([r.latency_ms for r in results if r.error is None]) or [0.0]

    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]

    return RunSummary(
        total=total,
        cache_hits=len(hits),
        keyword_matches_on_hits=len(keyword_hits),
        cache_hit_rate=(len(hits) / total) if total else 0.0,
        keyword_precision_on_hits=(len(keyword_hits) / len(hits)) if hits else 0.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
    )


def _print_table(results: list[QuestionResult], summary: RunSummary) -> None:
    print(
        f"\n{'hit':<4} {'ok':<3} {'ms':>6}  question"
    )
    print("-" * 80)
    for r in results:
        marker = "✓" if r.cache_hit else " "
        ok = "✓" if r.keyword_match else "✗"
        if r.error:
            ok = "E"
        print(f" {marker:<3} {ok:<3} {r.latency_ms:>6.0f}  {r.text[:60]}")
    print("-" * 80)
    print(f"total:                 {summary.total}")
    print(f"cache hits:            {summary.cache_hits} ({summary.cache_hit_rate:.0%})")
    print(
        f"keyword precision:     {summary.keyword_matches_on_hits}/"
        f"{summary.cache_hits} ({summary.keyword_precision_on_hits:.0%})"
    )
    print(f"p50 / p95 latency:     {summary.p50_latency_ms:.0f} ms / {summary.p95_latency_ms:.0f} ms")


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run the BOOTH eval harness.")
    parser.add_argument(
        "--testset",
        type=Path,
        default=Path(__file__).parent / "testsets" / "starter.yaml",
        help="Path to the YAML test set.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()
    return run(args.testset, args.out)


if __name__ == "__main__":
    sys.exit(_main())
