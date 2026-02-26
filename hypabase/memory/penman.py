"""PENMAN notation parser and sentence generator for Hypabase Memory.

Parses a compact S-expression-like format into Atom dataclasses,
and generates natural English sentences for embedding text.

Grammar:
    input  := atom+
    atom   := '(' WORD slot* ')'
    slot   := ':' WORD value
    value  := QUOTED_STRING | WORD | atom
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

# ---------------------------------------------------------------------------
# Atom dataclass
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    """A single parsed PENMAN atom (verb + slots)."""

    verb: str
    roles: list[tuple[str, str | Atom]] = field(default_factory=list)
    modifiers: dict[str, str | float | bool] = field(default_factory=dict)
    contexts: list[tuple[str, str | Atom]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------


class PenmanParseError(Exception):
    """Raised when PENMAN notation cannot be parsed."""

    def __init__(self, message: str, position: int = -1, source_text: str = "") -> None:
        self.position = position
        self.source_text = source_text
        if position >= 0:
            super().__init__(f"{message} (at position {position})")
        else:
            super().__init__(message)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class _TT(Enum):
    LPAREN = auto()
    RPAREN = auto()
    COLON = auto()
    WORD = auto()
    QUOTED_STRING = auto()
    EOF = auto()


@dataclass
class _Token:
    type: _TT
    value: str
    pos: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in (" ", "\t", "\n", "\r"):
            i += 1
        elif c == "(":
            tokens.append(_Token(_TT.LPAREN, "(", i))
            i += 1
        elif c == ")":
            tokens.append(_Token(_TT.RPAREN, ")", i))
            i += 1
        elif c == ":":
            tokens.append(_Token(_TT.COLON, ":", i))
            i += 1
        elif c == '"':
            # Quoted string
            start = i
            i += 1
            parts: list[str] = []
            _ESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    parts.append(_ESCAPE_MAP.get(text[i + 1], text[i + 1]))
                    i += 2
                else:
                    parts.append(text[i])
                    i += 1
            if i >= n:
                raise PenmanParseError("Unterminated quoted string", start, text)
            i += 1  # skip closing quote
            tokens.append(_Token(_TT.QUOTED_STRING, "".join(parts), start))
        else:
            # Word — runs until whitespace or special chars
            start = i
            while i < n and text[i] not in (" ", "\t", "\n", "\r", "(", ")", ":", '"'):
                i += 1
            tokens.append(_Token(_TT.WORD, text[start:i], start))
    tokens.append(_Token(_TT.EOF, "", n))
    return tokens


# ---------------------------------------------------------------------------
# Slot routing
# ---------------------------------------------------------------------------

ROLE_KEYS = frozenset({"subject", "object", "instrument", "recipient", "origin", "locus", "attribute", "value"})
MODIFIER_KEYS = frozenset({"tense", "mood", "negated", "memory_type", "importance"})
CONTEXT_KEYS = frozenset({"cause", "purpose", "condition"})


def _coerce_modifier(key: str, raw: str) -> str | float | bool:
    """Coerce a modifier string value to its appropriate type."""
    if key == "importance":
        try:
            return float(raw)
        except ValueError:
            return raw
    if key == "negated":
        return raw.lower() in ("true", "yes", "1")
    return raw


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[_Token], source: str) -> None:
        self._tokens = tokens
        self._source = source
        self._pos = 0

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: _TT) -> _Token:
        tok = self._advance()
        if tok.type != tt:
            raise PenmanParseError(
                f"Expected {tt.name}, got {tok.type.name} ({tok.value!r})",
                tok.pos,
                self._source,
            )
        return tok

    def parse(self) -> list[Atom]:
        atoms: list[Atom] = []
        while self._peek().type != _TT.EOF:
            atoms.append(self._parse_atom())
        if not atoms:
            raise PenmanParseError("Empty input — expected at least one atom", 0, self._source)
        return atoms

    def _parse_atom(self) -> Atom:
        self._expect(_TT.LPAREN)
        verb_tok = self._peek()
        if verb_tok.type not in (_TT.WORD, _TT.QUOTED_STRING):
            raise PenmanParseError(
                f"Expected verb after '(', got {verb_tok.type.name}",
                verb_tok.pos,
                self._source,
            )
        verb = self._advance().value

        roles: list[tuple[str, str | Atom]] = []
        modifiers: dict[str, str | float | bool] = {}
        contexts: list[tuple[str, str | Atom]] = []

        while self._peek().type == _TT.COLON:
            self._advance()  # consume ':'
            key_tok = self._peek()
            if key_tok.type not in (_TT.WORD, _TT.QUOTED_STRING):
                raise PenmanParseError(
                    f"Expected slot name after ':', got {key_tok.type.name}",
                    key_tok.pos,
                    self._source,
                )
            key = self._advance().value

            # Parse value
            val_tok = self._peek()
            if val_tok.type == _TT.LPAREN:
                value: str | Atom = self._parse_atom()
            elif val_tok.type in (_TT.WORD, _TT.QUOTED_STRING):
                value = self._advance().value
            else:
                raise PenmanParseError(
                    f"Expected value for :{key}, got {val_tok.type.name}",
                    val_tok.pos,
                    self._source,
                )

            # Route to appropriate category
            if key in ROLE_KEYS:
                roles.append((key, value))
            elif key in MODIFIER_KEYS:
                if isinstance(value, Atom):
                    raise PenmanParseError(
                        f"Modifier :{key} cannot have a nested atom as value",
                        val_tok.pos,
                        self._source,
                    )
                modifiers[key] = _coerce_modifier(key, value)
            elif key in CONTEXT_KEYS:
                contexts.append((key, value))
            else:
                # Unknown keys go to roles by default
                roles.append((key, value))

        self._expect(_TT.RPAREN)
        return Atom(verb=verb, roles=roles, modifiers=modifiers, contexts=contexts)


# ---------------------------------------------------------------------------
# Public parse function
# ---------------------------------------------------------------------------


def parse_penman(text: str) -> list[Atom]:
    """Parse PENMAN notation into a list of Atoms.

    Args:
        text: One or more PENMAN atoms, e.g.
            ``(prefers :subject Alice :object Python :memory_type semantic)``

    Returns:
        List of Atom dataclasses.

    Raises:
        PenmanParseError: On syntax errors, with position info.
    """
    if not text or not text.strip():
        raise PenmanParseError("Empty input — expected at least one atom", 0, text or "")
    tokens = _tokenize(text)
    parser = _Parser(tokens, text)
    return parser.parse()


# ---------------------------------------------------------------------------
# Sentence generator
# ---------------------------------------------------------------------------

# Preposition map for role → English connector
_ROLE_PREPOSITIONS: dict[str, str] = {
    "recipient": "to",
    "instrument": "with",
    "origin": "from",
    "locus": "at",
}

_CONTEXT_CONNECTORS: dict[str, str] = {
    "cause": "because",
    "purpose": "in order to",
    "condition": "if",
}


def _value_to_str(val: str | Atom) -> str:
    """Convert a role value to a string — recurse for nested atoms."""
    if isinstance(val, Atom):
        return atom_to_sentence(val)
    return val


def atom_to_sentence(atom: Atom) -> str:
    """Generate a natural English sentence from an Atom.

    Used to produce embedding text for semantic search.

    Args:
        atom: A parsed Atom.

    Returns:
        A human-readable sentence.
    """
    # Collect role values by role name
    subjects: list[str] = []
    objects: list[str] = []
    attributes: list[str] = []
    values: list[str] = []
    other_parts: list[str] = []

    for role_name, role_val in atom.roles:
        s = _value_to_str(role_val)
        if role_name == "subject":
            subjects.append(s)
        elif role_name == "object":
            objects.append(s)
        elif role_name == "attribute":
            attributes.append(s)
        elif role_name == "value":
            values.append(s)
        else:
            prep = _ROLE_PREPOSITIONS.get(role_name, role_name)
            other_parts.append(f"{prep} {s}")

    # Build sentence
    parts: list[str] = []

    subject_str = " and ".join(subjects) if subjects else ""

    # Special pattern for attribute+value with has/is verb
    if attributes and values and atom.verb in ("has", "is"):
        attr_str = " ".join(attributes)
        val_str = " ".join(values)
        if atom.verb == "is" and subject_str:
            # "the type of Python is programming language" reads better than
            # "Python is type programming language"
            parts.append(f"the {attr_str} of {subject_str}")
            parts.append("is")
            parts.append(val_str)
        elif atom.verb == "has" and subject_str:
            # "quick sort has time complexity O(n log n)"
            parts.append(subject_str)
            parts.append("has")
            parts.append(attr_str)
            parts.append(val_str)
        else:
            if subject_str:
                parts.append(subject_str)
            parts.append(atom.verb)
            parts.append(attr_str)
            parts.append(val_str)
    else:
        if subject_str:
            parts.append(subject_str)
        parts.append(atom.verb)
        if objects:
            parts.append(" and ".join(objects))
        if attributes:
            parts.append(" ".join(attributes))
        if values:
            parts.append(" ".join(values))

    parts.extend(other_parts)

    # Append context clauses
    for ctx_name, ctx_val in atom.contexts:
        connector = _CONTEXT_CONNECTORS.get(ctx_name, ctx_name)
        parts.append(f"{connector} {_value_to_str(ctx_val)}")

    return " ".join(parts)
