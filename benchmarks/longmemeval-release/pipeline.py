"""Per-question orchestration: ingest -> consolidate -> query -> judge."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import (
    ANSWERING_PROMPT,
    INGESTION_PROMPT,
    call_with_retry,
    get_answering_agent,
    get_ingestion_agent,
    get_judge_agent,
    get_judge_prompt,
)
from pydantic_ai.messages import ThinkingPart

from hypabase import Hypabase
from hypabase.engine.embeddings import EmbeddingProvider
from hypabase.memory import Memory

GRAPHS_DIR = Path(__file__).parent / "graphs"
RECALL_LIMIT = 50

logger = logging.getLogger(__name__)


def _extract_thinking(messages: list | None) -> str | None:
    """Extract thinking content from pydantic-ai message objects."""
    if not messages:
        return None
    parts = []
    for msg in messages:
        for part in getattr(msg, "parts", []):
            if isinstance(part, ThinkingPart) and part.content:
                parts.append(part.content)
    return "\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Session formatting
# ---------------------------------------------------------------------------


def format_exchange(exchange: list[dict]) -> str:
    """Format a single exchange (user + assistant pair) into text."""
    parts = []
    for turn in exchange:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _group_exchanges(session: list[dict]) -> list[list[dict]]:
    """Group session turns into exchanges (user + assistant pairs)."""
    exchanges: list[list[dict]] = []
    i = 0
    while i < len(session):
        turn = session[i]
        exchange = [turn]
        if i + 1 < len(session) and session[i + 1].get("role") == "assistant" and turn.get("role") == "user":
            exchange.append(session[i + 1])
            i += 2
        else:
            i += 1
        exchanges.append(exchange)
    return exchanges


# ---------------------------------------------------------------------------
# Phase 1: Ingestion (adjacent windows, S&B zero-overlap principle)
# ---------------------------------------------------------------------------

WINDOW_SIZE = 3  # exchanges per non-overlapping ingestion window


def _parse_session_date(date_str: str) -> float | None:
    """Parse a session date string to a unix timestamp."""
    for fmt in ["%Y/%m/%d (%a) %H:%M", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def ingest_sessions(
    mem: Memory,
    sessions: list[list[dict]],
    session_dates: list[str],
) -> dict[str, Any]:
    """Ingest all sessions into Memory via adjacent-window LLM extraction.

    S&B zero-overlap principle: each exchange appears in exactly one
    ingestion window. Eliminates duplicate extraction from overlapping
    context windows while preserving coreference context within each group.
    """
    agent = get_ingestion_agent()
    total_stored = 0
    parse_errors = 0
    llm_errors: list[str] = []
    exchange_logs: list[dict] = []

    for si, session in enumerate(sessions):
        session_date = session_dates[si] if si < len(session_dates) else "unknown"
        exchanges = _group_exchanges(session)

        # Adjacent windows: non-overlapping groups of WINDOW_SIZE exchanges
        for gi in range(0, len(exchanges), WINDOW_SIZE):
            group = exchanges[gi : gi + WINDOW_SIZE]
            conversation = "\n\n".join(format_exchange(ex) for ex in group)
            if not conversation.strip():
                continue

            prompt = INGESTION_PROMPT.format(
                session_date=session_date,
                conversation=conversation,
            )

            log_entry: dict[str, Any] = {
                "session": si,
                "group": gi // WINDOW_SIZE,
                "exchanges": f"{gi}-{gi + len(group) - 1}",
                "prompt": prompt,
                "llm_output": None,
                "atoms_stored": 0,
                "error": None,
            }
            t0 = time.monotonic()

            penman_text, err, _ = call_with_retry(agent, prompt)
            if err:
                llm_errors.append(f"session_{si}_group_{gi}: {err}")
                log_entry["error"] = err
                log_entry["elapsed_s"] = round(time.monotonic() - t0, 2)
                exchange_logs.append(log_entry)
                continue

            log_entry["llm_output"] = penman_text

            if not penman_text or not penman_text.strip():
                log_entry["elapsed_s"] = round(time.monotonic() - t0, 2)
                exchange_logs.append(log_entry)
                continue

            try:
                created_ts = _parse_session_date(session_date)
                result = mem.remember(
                    penman=penman_text,
                    source=session_date,
                    confidence=0.9,
                    created_at=created_ts,
                )
                stored = result["stored"]
                total_stored += stored
                log_entry["atoms_stored"] = stored
            except Exception as e:
                parse_errors += 1
                log_entry["error"] = str(e)
                logger.debug("PENMAN error: %s — text: %s", e, penman_text[:80])

            log_entry["elapsed_s"] = round(time.monotonic() - t0, 2)
            exchange_logs.append(log_entry)

    return {
        "atoms_stored": total_stored,
        "parse_errors": parse_errors,
        "llm_errors": llm_errors,
        "sessions_processed": len(sessions),
        "exchange_logs": exchange_logs,
    }


# ---------------------------------------------------------------------------
# Phase 2: Query
# ---------------------------------------------------------------------------


def query_memory(
    mem: Memory,
    question: str,
) -> tuple[str, int, dict]:
    """Query Memory via integrated recall — FTS5 lexical + cosine semantic.

    Single call to recall(query=Q) runs both pathways in parallel and
    interleaves the results (CLS dual-route model). No separate entity
    pass needed — FTS5 keyword matching finds entity-connected edges
    via word-form matching on node names in the edge index.

    Returns (formatted_memories, count, query_trace).
    """
    t0 = time.monotonic()

    results = mem.recall(query=question, limit=RECALL_LIMIT)

    if not results:
        query_trace = {
            "total_results": 0,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
        return "(No memories found)", 0, query_trace

    # Format memories as bullet points
    parts = []
    for i, r in enumerate(results, 1):
        text = r.get("text", "")

        mt = r.get("memory_type", "")
        mood = r.get("mood", "actual")
        negated = r.get("negated", False)

        line = f"- {text}"

        extras = []
        if mt:
            extras.append(mt)
        if mood != "actual":
            extras.append(f"mood: {mood}")
        if negated:
            extras.append("NEGATED")

        edge = r.get("edge")
        if edge:
            tense = edge.properties.get("tense")
            if tense:
                extras.append(f"tense: {tense}")
            if edge.source:
                extras.append(f"recorded: {edge.source}")

        if extras:
            line += f" ({', '.join(extras)})"
        parts.append(line)

    query_trace = {
        "total_results": len(results),
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    return "\n".join(parts), len(results), query_trace


# ---------------------------------------------------------------------------
# Phase 3: Judge
# ---------------------------------------------------------------------------


def judge_answer(
    question: str,
    ground_truth: str,
    generated: str,
    question_type: str,
    question_id: str,
) -> tuple[bool, str, dict]:
    """Judge using official LongMemEval per-type prompts. Returns (is_correct, explanation, judge_trace)."""
    prompt = get_judge_prompt(question_type, question_id, question, ground_truth, generated)
    t0 = time.monotonic()
    result_text, err, _ = call_with_retry(get_judge_agent(), prompt)
    elapsed = round(time.monotonic() - t0, 2)
    judge_trace = {"judge_prompt": prompt, "elapsed_s": elapsed}
    if err:
        return False, f"Judge error: {err}", judge_trace
    is_correct = "yes" in result_text.lower()
    return is_correct, result_text, judge_trace


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_question(
    item: dict,
    embedder: EmbeddingProvider | None = None,
    recall_fn=None,
) -> dict:
    """Execute the full ingest -> consolidate -> query -> judge pipeline for one question."""
    question_id = str(item.get("question_id", ""))
    question_type = item["question_type"]
    question = item["question"]
    ground_truth = str(item.get("answer", ""))
    sessions = item.get("haystack_sessions", [])
    question_date = item.get("question_date", "")

    # Session dates for temporal context
    session_dates = item.get("haystack_dates", [])
    if not session_dates:
        session_dates = [question_date] * len(sessions)

    result: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "ground_truth": ground_truth,
    }

    trace: dict[str, Any] = {}
    timing: dict[str, float] = {}
    t_total = time.monotonic()

    # Persist graph to disk for later inspection
    GRAPHS_DIR.mkdir(exist_ok=True)
    db_path = GRAPHS_DIR / f"{question_id}.db"
    if db_path.exists():
        db_path.unlink()

    hb: Hypabase | None = None
    try:
        hb = Hypabase(str(db_path), embedder=embedder)
        mem = Memory(hb=hb, embedder=embedder)

        # Phase 1: Ingest sessions
        t0 = time.monotonic()
        ingest_diag = ingest_sessions(mem, sessions, session_dates)
        timing["ingest_s"] = round(time.monotonic() - t0, 2)
        result["ingest"] = {k: v for k, v in ingest_diag.items() if k != "exchange_logs"}
        trace["ingestion"] = ingest_diag.get("exchange_logs", [])

        # Consolidate entities after ingestion
        if ingest_diag["atoms_stored"] > 0:
            try:
                mem.consolidate()
            except Exception as e:
                logger.debug("Consolidation error: %s", e)

        # Phase 2: Query + Answer
        t0 = time.monotonic()
        if recall_fn is not None:
            memories_text, mem_count, query_trace = recall_fn(hb, embedder, question, RECALL_LIMIT)
        else:
            memories_text, mem_count, query_trace = query_memory(mem, question)
        timing["query_s"] = round(time.monotonic() - t0, 2)
        result["memories_retrieved_count"] = mem_count
        result["memories_text"] = memories_text
        trace["query"] = query_trace

        answering_prompt = ANSWERING_PROMPT.format(
            question_date=question_date,
            memories=memories_text,
            question=question,
        )
        trace["answering_prompt"] = answering_prompt

        t0 = time.monotonic()
        generated, err, answer_msgs = call_with_retry(get_answering_agent(), answering_prompt)
        timing["answer_s"] = round(time.monotonic() - t0, 2)
        thinking = _extract_thinking(answer_msgs)
        if thinking:
            trace["answering_thinking"] = thinking
        if err:
            result["error"] = f"Answering error: {err}"
            result["correct"] = False
            result["generated_answer"] = ""
            return result

        result["generated_answer"] = generated

        # Phase 3: Judge
        is_correct, judge_explanation, judge_trace = judge_answer(
            question,
            ground_truth,
            generated,
            question_type,
            question_id,
        )
        timing["judge_s"] = judge_trace["elapsed_s"]
        result["correct"] = is_correct
        result["judge_explanation"] = judge_explanation
        trace["judge_prompt"] = judge_trace["judge_prompt"]

    except Exception as e:
        result["error"] = str(e)
        result["correct"] = False
        result["generated_answer"] = result.get("generated_answer", "")
    finally:
        if hb is not None:
            hb.close()

    timing["total_s"] = round(time.monotonic() - t_total, 2)
    trace["timing"] = timing
    result["trace"] = trace

    return result


def run_question_recall_only(
    item: dict,
    embedder: EmbeddingProvider | None = None,
    recall_fn=None,
) -> dict:
    """Recall-only pipeline: load persisted graph -> query -> answer -> judge.

    Skips ingestion and consolidation. Requires a pre-built graph DB
    in GRAPHS_DIR for the given question_id.
    """
    question_id = str(item.get("question_id", ""))
    question_type = item["question_type"]
    question = item["question"]
    ground_truth = str(item.get("answer", ""))
    question_date = item.get("question_date", "")

    result: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "question": question,
        "ground_truth": ground_truth,
    }

    trace: dict[str, Any] = {}
    timing: dict[str, float] = {}
    t_total = time.monotonic()

    db_path = GRAPHS_DIR / f"{question_id}.db"
    if not db_path.exists():
        result["error"] = f"Graph DB not found: {db_path}"
        result["correct"] = False
        result["generated_answer"] = ""
        return result

    hb: Hypabase | None = None
    try:
        hb = Hypabase(str(db_path), embedder=embedder)

        # Phase 1: Query + Answer (skip ingestion)
        t0 = time.monotonic()
        if recall_fn is not None:
            memories_text, mem_count, query_trace = recall_fn(hb, embedder, question, RECALL_LIMIT)
        else:
            mem = Memory(hb=hb, embedder=embedder)
            memories_text, mem_count, query_trace = query_memory(mem, question)
        timing["query_s"] = round(time.monotonic() - t0, 2)
        result["memories_retrieved_count"] = mem_count
        result["memories_text"] = memories_text
        trace["query"] = query_trace

        answering_prompt = ANSWERING_PROMPT.format(
            question_date=question_date,
            memories=memories_text,
            question=question,
        )
        trace["answering_prompt"] = answering_prompt

        t0 = time.monotonic()
        generated, err, answer_msgs = call_with_retry(get_answering_agent(), answering_prompt)
        timing["answer_s"] = round(time.monotonic() - t0, 2)
        thinking = _extract_thinking(answer_msgs)
        if thinking:
            trace["answering_thinking"] = thinking
        if err:
            result["error"] = f"Answering error: {err}"
            result["correct"] = False
            result["generated_answer"] = ""
            return result

        result["generated_answer"] = generated

        # Phase 2: Judge
        is_correct, judge_explanation, judge_trace = judge_answer(
            question,
            ground_truth,
            generated,
            question_type,
            question_id,
        )
        timing["judge_s"] = judge_trace["elapsed_s"]
        result["correct"] = is_correct
        result["judge_explanation"] = judge_explanation
        trace["judge_prompt"] = judge_trace["judge_prompt"]

    except Exception as e:
        result["error"] = str(e)
        result["correct"] = False
        result["generated_answer"] = result.get("generated_answer", "")
    finally:
        if hb is not None:
            hb.close()

    timing["total_s"] = round(time.monotonic() - t_total, 2)
    trace["timing"] = timing
    result["trace"] = trace

    return result
