"""Dataset loading and balanced sampling for LongMemEval."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"

HF_REPO = "xiaowu0162/longmemeval-cleaned"
ORACLE_FILENAME = "longmemeval_oracle.json"


def load_oracle_data(cache_dir: Path | None = None) -> list[dict]:
    """Load LongMemEval oracle dataset, downloading from HuggingFace if needed."""
    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    oracle_path = cache_dir / ORACLE_FILENAME

    if not oracle_path.exists():
        from huggingface_hub import hf_hub_download

        print(f"Downloading {ORACLE_FILENAME} from {HF_REPO}...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=ORACLE_FILENAME,
            local_dir=str(cache_dir),
            repo_type="dataset",
        )

    with open(oracle_path) as f:
        data = json.load(f)

    return data


def sample_balanced(
    data: list[dict],
    n_total: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Sample n_total questions proportionally across question types.

    Preserves the original distribution of question types in the dataset.
    Uses largest-remainder allocation to distribute any rounding surplus.
    """
    rng = random.Random(seed)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_type[item["question_type"]].append(item)

    # Shuffle within each type
    for items in by_type.values():
        rng.shuffle(items)

    # Proportional allocation (largest-remainder method)
    total = len(data)
    quotas: dict[str, float] = {qt: len(items) / total * n_total for qt, items in by_type.items()}
    allocation: dict[str, int] = {qt: int(q) for qt, q in quotas.items()}
    remainders = sorted(quotas.keys(), key=lambda qt: quotas[qt] - allocation[qt], reverse=True)
    shortfall = n_total - sum(allocation.values())
    for qt in remainders[:shortfall]:
        allocation[qt] += 1

    sampled: list[dict] = []
    for qt in sorted(by_type):
        items = by_type[qt]
        take = min(len(items), allocation.get(qt, 0))
        sampled.extend(items[:take])

    return sampled[:n_total]


def describe_sample(data: list[dict]) -> dict[str, int]:
    """Return question type distribution."""
    counts: dict[str, int] = defaultdict(int)
    for item in data:
        counts[item["question_type"]] += 1
    return dict(sorted(counts.items()))
