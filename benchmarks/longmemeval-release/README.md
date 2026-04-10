# LongMemEval Benchmark

Evaluates Hypabase Memory on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025), a 500-question benchmark for long-term memory in conversational AI.

## Pipeline

1. **Ingest** -- Extract facts from conversation sessions as AMR graphs in PENMAN notation, stored as hyperedges in Hypabase
2. **Consolidate** -- Merge duplicate entity nodes via cosine similarity
3. **Recall** -- QB-PPR (Query-Biased Personalized PageRank) on the hypergraph fused with cosine KNN via Reciprocal Rank Fusion
4. **Answer** -- LLM generates answer from retrieved memories
5. **Judge** -- LLM judges correctness using official LongMemEval per-type prompts

## Results

| Category | N | Accuracy |
|---|---|---|
| **Overall** | **500** | **87.4%** |
| single-session-preference | 30 | 100.0% |
| single-session-user | 70 | 92.9% |
| single-session-assistant | 56 | 89.3% |
| knowledge-update | 78 | 88.5% |
| temporal-reasoning | 133 | 87.2% |
| multi-session | 133 | 80.5% |

Task-averaged accuracy: 89.7%

## Requirements

- Python 3.11+
- AWS Bedrock credentials (for Claude Sonnet 4.6 ingestion, Claude Opus 4.6 answering/judging)
- ~2GB disk for graph databases (generated during ingestion)

## Usage

```bash
cd hypabase/benchmarks/longmemeval-release

# Install dependencies
uv sync

# Run full benchmark (500 questions, ~2-3 hours)
uv run python run.py --workers 5 --output results.json

# Smaller test run
uv run python run.py --n 10 --seed 42 --workers 1 --output test.json

# Recall-only (skip ingestion, reuse persisted graphs)
uv run python run.py --recall-only --from-results results.json --output recall_results.json
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--n` | 500 | Number of questions to evaluate |
| `--seed` | 42 | Random seed for balanced sampling |
| `--workers` | 5 | Parallel workers |
| `--output` | `longmemeval_results.json` | Output file |
| `--algorithm` | `qbppr` | Recall algorithm: `qbppr` (QB-PPR + cosine RRF) or `memory` (baseline Memory.recall) |
| `--recall-only` | off | Skip ingestion, load persisted graphs |
| `--from-results` | | Load question IDs from a previous results file |
| `--ids` | | Run only specific question IDs |
| `--no-embed` | off | Disable embeddings |

## Algorithm: QB-PPR

Query-Biased Personalized PageRank on hypergraph (Chitra & Raphael 2019) with three corrections for star-topology personal memory graphs:

1. **Inverse-degree vertex weights** -- Suppresses hub nodes (e.g. "user"), amplifies specific entities
2. **Non-linear edge weight amplification** -- cosine^4 stretches similarity ratios for meaningful walk bias
3. **High teleportation** -- alpha=0.5 keeps mass near query-relevant seeds on small graphs

Combined with cosine KNN via Reciprocal Rank Fusion (k=60).
