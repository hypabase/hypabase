"""Metrics calculation and display for LongMemEval benchmark."""

from __future__ import annotations

from collections import defaultdict


def compute_metrics(results: list[dict]) -> dict:
    """Compute per-type and overall accuracy from question results."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    n_total = len(results)
    n_correct = sum(1 for r in results if r.get("correct", False))
    n_errors = sum(1 for r in results if "error" in r)

    overall = {
        "n": n_total,
        "correct": n_correct,
        "accuracy": n_correct / n_total if n_total > 0 else 0,
        "errors": n_errors,
    }

    per_type: dict[str, dict] = {}
    for qt in sorted(by_type):
        typed = by_type[qt]
        tn = len(typed)
        tc = sum(1 for r in typed if r.get("correct", False))
        per_type[qt] = {
            "n": tn,
            "correct": tc,
            "accuracy": tc / tn if tn > 0 else 0,
        }

    return {"overall": overall, "per_type": per_type}


def print_results(metrics: dict) -> None:
    """Print a formatted results table."""
    overall = metrics["overall"]
    per_type = metrics["per_type"]

    print(f"\n{'=' * 70}")
    print("  LONGMEMEVAL BENCHMARK -- Hypabase Memory")
    print(f"{'=' * 70}")
    print(f"  Overall: {overall['accuracy']:.1%} ({overall['correct']}/{overall['n']})")
    if overall["errors"]:
        print(f"  Errors:  {overall['errors']}")

    print(f"\n  {'Question Type':<35} {'N':>4} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-' * 60}")
    for qt, m in sorted(per_type.items()):
        print(f"  {qt:<35} {m['n']:>4} {m['correct']:>8} {m['accuracy']:>9.1%}")
    print()
