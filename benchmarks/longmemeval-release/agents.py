"""Pydantic-AI agents for the LongMemEval benchmark pipeline."""

from __future__ import annotations

import time

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModelSettings

INGESTION_MODEL = "bedrock:global.anthropic.claude-sonnet-4-6"
ANSWERING_MODEL = "bedrock:global.anthropic.claude-opus-4-6-v1"
JUDGE_MODEL = "bedrock:global.anthropic.claude-opus-4-6-v1"
MODEL = ANSWERING_MODEL  # exported for display in run.py

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

INGESTION_SYSTEM = """\
You read conversations between a human and an assistant and extract the \
information as AMR graphs in PENMAN notation for a memory graph.

WHAT TO EXTRACT
---------------
Every detail mentioned is information. Subordinate clauses, side remarks, \
and casual mentions carry facts just as important as the main topic. \
"I'm fixing my desk, which I built with my dad last summer" contains \
the fixing intent AND the origin AND the timing — all must be captured. \
Never discard a detail because it was not the focus of the sentence.

Extract what both speakers stated. The assistant's answers, recommendations, \
and specific names are information — extract them just as you would the \
user's own statements. Note negations and corrections.

HOW TO WRITE
------------
Following AMR principles: one sentence = one AMR. All semantic content \
from a sentence — main clause, subordinate clauses, relative clauses, \
temporal references — belongs in one PENMAN graph. Do not decompose a \
sentence into separate graphs for each detail.

A single well-filled graph with many role edges is always better than \
several sparse ones. Pack who, what, when, where, from whom, how much, \
and why into one graph.

Preserve every specific detail exactly as stated: names, numbers, dates, \
quantities, sources, companions, locations. Never drop these for a vaguer \
description. Relative time references ("last week", "two months ago", \
"yesterday") are facts — include them in the :locus role.

Resolve all pronouns and references to their entity name. If the \
conversation says "my cat" earlier and then "she", use "cat" not "she". \
Use the same canonical name each time an entity appears.

Write AMR graphs in PENMAN notation: (verb :role entity ...)

Each graph is a predicate (verb) with participants in role edges:

    (verb :role "entity" :role "entity" ...)

ROLES (fill in what applies, skip what doesn't)
-----
:subject     who or what it's about
:object      what is acted on
:recipient   who receives or benefits
:instrument  tool, method, or means used
:origin      where it came from, who gave it, previous state
:locus       where, when, or in what context
:attribute   a named property or dimension
:value       the specific value of that property

MODIFIERS (metadata about the fact)
---------
:tense        past, present, future
:mood         actual (default), planned, uncertain, normative, conditional
:negated      true or false
:memory_type  episodic (events), semantic (facts), procedural (how-to)
:importance   0.0 to 1.0

CONTEXT (why, for what, under what condition -- can be nested)
-------
:cause       why it happened
:purpose     what for
:condition   if/when/unless

NESTING
-------
Any role can hold a nested graph instead of a string:

    (believes :subject Alice :object (is :subject deadline :value Friday))

EXAMPLES
--------
Complex sentence — one AMR, all details preserved:
    "Alice is repainting her bookshelf, which she bought from a thrift \
store two weeks ago."
    (repaints :subject Alice :object "bookshelf" :origin "thrift store"
     :locus "two weeks ago" :tense present :memory_type episodic)

    WRONG — decomposing into sparse graphs loses the connection:
    (has :subject Alice :object "bookshelf")
    (repaints :subject Alice :object "bookshelf")

Event with time, place, companions, and cost:
    "Bob had lunch with Carol at the diner on Tuesday for $25."
    (dined :subject Bob :object "diner" :locus Tuesday
     :instrument Carol :value "$25" :tense past :memory_type episodic)

Assistant recommendation with specifics:
    (recommended :subject assistant :object "project management tools"
     :recipient Alice :value "Trello for kanban, Notion for docs"
     :memory_type semantic)

Property:
    (has :subject "quick sort" :attribute "time complexity"
     :value "O(n log n)" :memory_type semantic)

Negation:
    (uses :subject Django :object "Python 2" :negated true
     :memory_type semantic)

ENTITY NAMING
-------------
Use the same name each time you refer to an entity. "Alice Smith" and
"alice smith" are the same entity (case-insensitive), but "Alice" and
"Alice Smith" are different entities until consolidate() merges them.
Pick one canonical name per entity and reuse it across all memories.
Use full descriptive names: "machine learning" not "ML", "JavaScript" not "JS".
Always refer to the human speaker as "user".

Output ONLY PENMAN notation. No explanations, no numbering, no markdown."""

ANSWERING_SYSTEM = """\
You are a helpful assistant answering questions using the user's stored memories.

INSTRUCTIONS
- Memories are ordered by relevance — earlier memories are more likely to \
contain the answer, but always check all of them.
- Use the retrieved memories as context about the user's preferences, \
experiences, and situation.
- For factual recall questions, answer strictly from the provided context.
- The user's stored preferences apply broadly — if they liked something in \
one context, apply that preference to similar situations. Lead with what \
you know about the user, not generic advice.
- Each memory has a recorded timestamp — the date and time it was recorded. \
Use these to resolve relative time references like "last week" or "two months ago".
- When values conflict, use the most recently recorded memory.
- If the context contains relevant information, use it to answer -- even if \
it requires inference.
- Only say information is unavailable when the context has truly nothing \
relevant -- no related preferences, no related facts, nothing applicable.

OUTPUT FORMAT
Respond directly with just the answer. No preamble, no "Based on...", no \
"According to..."."""

# ---------------------------------------------------------------------------
# Judge templates -- verbatim from LongMemEval evaluate_qa.py
# (xiaowu0162/LongMemEval, ICLR 2025)
# ---------------------------------------------------------------------------

JUDGE_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {ground_truth}\n\nModel Response: "
    "{generated_answer}\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
    "response is still correct. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {ground_truth}\n\nModel Response: "
    "{generated_answer}\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer."
    "\n\nQuestion: {question}\n\nCorrect Answer: {ground_truth}\n\nModel Response: "
    "{generated_answer}\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal "
    "information correctly."
    "\n\nQuestion: {question}\n\nRubric: {ground_truth}\n\nModel Response: "
    "{generated_answer}\n\nIs the model response correct? Answer yes or no only."
)

JUDGE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not."
    "\n\nQuestion: {question}\n\nExplanation: {ground_truth}\n\nModel Response: "
    "{generated_answer}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)

JUDGE_TEMPLATES: dict[str, str] = {
    "single-session-user": JUDGE_STANDARD,
    "single-session-assistant": JUDGE_STANDARD,
    "multi-session": JUDGE_STANDARD,
    "temporal-reasoning": JUDGE_TEMPORAL,
    "knowledge-update": JUDGE_KNOWLEDGE_UPDATE,
    "single-session-preference": JUDGE_PREFERENCE,
}


def get_judge_prompt(
    question_type: str,
    question_id: str,
    question: str,
    ground_truth: str,
    generated_answer: str,
) -> str:
    """Build the judge prompt matching LongMemEval's evaluate_qa.py exactly."""
    is_abstention = "_abs" in question_id
    if is_abstention:
        template = JUDGE_ABSTENTION
    else:
        template = JUDGE_TEMPLATES[question_type]
    return template.format(
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer,
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INGESTION_PROMPT = """\
Session date: {session_date}

Conversation:
{conversation}

Extract facts from these exchanges as AMR in PENMAN notation:"""

ANSWERING_PROMPT = """\
Today's date: {question_date}

Retrieved memories:
{memories}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Agent singletons
# ---------------------------------------------------------------------------

ANSWERING_SETTINGS = BedrockModelSettings()

_ingestion_agent: Agent | None = None
_answering_agent: Agent | None = None
_judge_agent: Agent | None = None


def get_ingestion_agent() -> Agent:
    global _ingestion_agent
    if _ingestion_agent is None:
        _ingestion_agent = Agent(INGESTION_MODEL, system_prompt=INGESTION_SYSTEM)
    return _ingestion_agent


def get_answering_agent() -> Agent:
    global _answering_agent
    if _answering_agent is None:
        _answering_agent = Agent(
            ANSWERING_MODEL,
            system_prompt=ANSWERING_SYSTEM,
            model_settings=ANSWERING_SETTINGS,
        )
    return _answering_agent


def get_judge_agent() -> Agent:
    global _judge_agent
    if _judge_agent is None:
        _judge_agent = Agent(JUDGE_MODEL)
    return _judge_agent


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------


def call_with_retry(
    agent: Agent,
    prompt: str,
    max_retries: int = 3,
) -> tuple[str | None, str | None, list | None]:
    """Call an agent with exponential backoff. Returns (output, error, new_messages)."""
    for attempt in range(max_retries):
        try:
            result = agent.run_sync(prompt)
            return result.output, None, result.new_messages()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return None, str(e), None
    return None, "max retries exhausted", None
