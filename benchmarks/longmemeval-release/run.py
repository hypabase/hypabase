"""LongMemEval benchmark for the Hypabase Memory system.

Usage:
    cd hypabase/benchmarks/longmemeval-release

    # Full pipeline (ingest + recall + answer + judge)
    uv run python run.py [--n 500] [--seed 42] [--workers 5] [--output results.json]

    # Recall-only (reuse persisted graphs, skip ingestion)
    uv run python run.py --recall-only --from-results longmemeval_results.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

# Thread-local storage for per-thread embedders
_thread_local = threading.local()


def _get_thread_embedder(use_embed: bool):
    """Get or create a per-thread FastEmbedProvider instance."""
    if not use_embed:
        return None
    if not hasattr(_thread_local, "embedder"):
        from hypabase.engine.embeddings import FastEmbedProvider

        _thread_local.embedder = FastEmbedProvider()
    return _thread_local.embedder


def _run_question_wrapper(item: dict, use_embed: bool, recall_only: bool = False, algorithm: str = "qbppr") -> dict:
    """Wrapper that creates a per-thread embedder and runs the question."""
    embedder = _get_thread_embedder(use_embed)

    recall_fn = None
    if algorithm != "memory":
        from algorithms import get_algorithm

        recall_fn = get_algorithm(algorithm)

    if recall_only:
        from pipeline import run_question_recall_only

        return run_question_recall_only(item, embedder=embedder, recall_fn=recall_fn)
    else:
        from pipeline import run_question

        return run_question(item, embedder=embedder, recall_fn=recall_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for Hypabase Memory")
    parser.add_argument("--n", type=int, default=500, help="Number of questions to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--no-embed", action="store_true", help="Disable embeddings (faster, less accurate recall)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers (default 5 to stay under rate limits)")
    parser.add_argument("--output", type=str, default="longmemeval_results.json", help="Output JSON file")
    parser.add_argument("--ids", type=str, nargs="+", help="Run only these question IDs")
    parser.add_argument(
        "--recall-only",
        action="store_true",
        help="Skip ingestion, use persisted graphs (recall + answer + judge only)",
    )
    parser.add_argument("--from-results", type=str, help="Load question IDs from a previous results JSON file")
    parser.add_argument(
        "--algorithm", type=str, default="qbppr", help="Recall algorithm: 'qbppr' (default QB-PPR + cosine RRF)"
    )
    args = parser.parse_args()

    from data import describe_sample, load_oracle_data, sample_balanced

    use_embed = not args.no_embed

    if use_embed:
        # Warm up the embedder on the main thread (downloads model if needed)
        try:
            from hypabase.engine.embeddings import FastEmbedProvider

            print("Initializing FastEmbed embedder...")
            _warmup = FastEmbedProvider()
            print("  Embedder ready.")
        except Exception as e:
            print(f"  Warning: Could not initialize embedder ({e}), running without embeddings.")
            use_embed = False

    # Load dataset
    print("Loading LongMemEval oracle dataset...")
    data = load_oracle_data()
    print(f"  Loaded {len(data)} questions.")

    # Filter or sample
    if args.from_results:
        with open(args.from_results) as _f:
            _prev = json.load(_f)
        prev_ids = [r["question_id"] for r in _prev["results"]]
        id_set = set(prev_ids)
        data = [item for item in data if str(item.get("question_id", "")) in id_set]
        # Preserve original order
        id_order = {qid: i for i, qid in enumerate(prev_ids)}
        data.sort(key=lambda item: id_order.get(str(item.get("question_id", "")), 999))
    elif args.ids:
        id_set = set(args.ids)
        data = [item for item in data if str(item.get("question_id", "")) in id_set]
    elif args.n < len(data):
        data = sample_balanced(data, n_total=args.n, seed=args.seed)
    else:
        data = data[: args.n]

    dist = describe_sample(data)
    print(f"  Sample ({len(data)} questions): {dist}")

    # Determine model from agents module
    from agents import MODEL

    recall_only = args.recall_only

    # Run benchmark
    mode = "recall-only (persisted graphs)" if recall_only else "full pipeline"
    print(f"\nRunning {len(data)} questions through Hypabase Memory pipeline...")
    print(f"  Mode: {mode}")
    print(f"  Model: {MODEL}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Embeddings: {'on' if use_embed else 'off'}")
    print(f"  Workers: {args.workers}")
    print()

    from tqdm import tqdm

    start = time.time()
    results: list[dict] = [None] * len(data)  # type: ignore[list-item]

    if args.workers <= 1:
        # Sequential execution
        for i, item in enumerate(tqdm(data, desc="Evaluating", file=sys.stderr)):
            result = _run_question_wrapper(item, use_embed, recall_only, args.algorithm)
            results[i] = result
            status = "OK" if result.get("correct") else "MISS"
            if "error" in result:
                status = "ERR"
            tqdm.write(
                f"  [{status}] {result['question_type']}: "
                f"Q={result['question'][:60]}... "
                f"A={result.get('generated_answer', '')[:40]}",
                file=sys.stderr,
            )
    else:
        # Parallel execution with ThreadPoolExecutor
        # Each thread gets its own embedder via thread-local storage
        pbar = tqdm(total=len(data), desc="Evaluating", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(_run_question_wrapper, item, use_embed, recall_only, args.algorithm): i
                for i, item in enumerate(data)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "question_id": str(data[idx].get("question_id", "")),
                        "question_type": data[idx]["question_type"],
                        "question": data[idx]["question"],
                        "ground_truth": str(data[idx].get("answer", "")),
                        "error": str(e),
                        "correct": False,
                        "generated_answer": "",
                    }
                results[idx] = result

                status = "OK" if result.get("correct") else "MISS"
                if "error" in result:
                    status = "ERR"
                tqdm.write(
                    f"  [{status}] {result['question_type']}: "
                    f"Q={result['question'][:60]}... "
                    f"A={result.get('generated_answer', '')[:40]}",
                    file=sys.stderr,
                )
                pbar.update(1)
        pbar.close()

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed / len(data):.1f}s per question)", file=sys.stderr)

    # Compute and display metrics
    from metrics import compute_metrics, print_results

    metrics = compute_metrics(results)
    print_results(metrics)

    # Save full results
    output_path = Path(__file__).parent / args.output
    save_data = {
        "config": {
            "n_questions": len(data),
            "seed": args.seed,
            "embeddings": use_embed,
            "recall_only": recall_only,
            "algorithm": args.algorithm,
            "model": MODEL,
            "workers": args.workers,
            "dataset": "longmemeval_oracle",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(elapsed, 1),
        },
        "metrics": metrics,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"Full results saved to {output_path}")


if __name__ == "__main__":
    main()
