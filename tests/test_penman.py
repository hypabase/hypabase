"""Tests for the PENMAN parser and sentence generator."""

from __future__ import annotations

import pytest

from hypabase.memory.penman import Atom, PenmanParseError, atom_to_sentence, parse_penman

# ==================================================================
# TestParse
# ==================================================================


class TestParse:
    def test_flat_two_roles(self):
        atoms = parse_penman("(prefers :subject Alice :object Python)")
        assert len(atoms) == 1
        atom = atoms[0]
        assert atom.verb == "prefers"
        assert atom.roles == [("subject", "Alice"), ("object", "Python")]

    def test_all_eight_roles(self):
        atoms = parse_penman(
            "(assigned :subject Alice :object task :instrument Jira "
            ":recipient Bob :origin backlog :locus Monday "
            ":attribute priority :value high)"
        )
        atom = atoms[0]
        assert atom.verb == "assigned"
        role_names = [r[0] for r in atom.roles]
        assert role_names == [
            "subject",
            "object",
            "instrument",
            "recipient",
            "origin",
            "locus",
            "attribute",
            "value",
        ]
        role_values = [r[1] for r in atom.roles]
        assert role_values == [
            "Alice",
            "task",
            "Jira",
            "Bob",
            "backlog",
            "Monday",
            "priority",
            "high",
        ]

    def test_modifiers(self):
        atoms = parse_penman(
            "(deploy :subject Alice :object API "
            ":tense past :mood planned :negated true "
            ":memory_type episodic :importance 0.8)"
        )
        atom = atoms[0]
        assert atom.modifiers["tense"] == "past"
        assert atom.modifiers["mood"] == "planned"
        assert atom.modifiers["negated"] is True
        assert atom.modifiers["memory_type"] == "episodic"
        assert atom.modifiers["importance"] == pytest.approx(0.8)

    def test_contexts(self):
        atoms = parse_penman(
            '(failed :subject server :object request :cause overload :purpose nothing :condition "high traffic")'
        )
        atom = atoms[0]
        assert len(atom.contexts) == 3
        assert atom.contexts[0] == ("cause", "overload")
        assert atom.contexts[1] == ("purpose", "nothing")
        assert atom.contexts[2] == ("condition", "high traffic")

    def test_nested_in_role(self):
        atoms = parse_penman("(believes :subject Alice :object (is :subject deadline :value Friday))")
        atom = atoms[0]
        assert atom.verb == "believes"
        assert atom.roles[0] == ("subject", "Alice")
        nested = atom.roles[1][1]
        assert isinstance(nested, Atom)
        assert nested.verb == "is"
        assert nested.roles == [("subject", "deadline"), ("value", "Friday")]

    def test_nested_in_context(self):
        atoms = parse_penman(
            '(crashed :subject server :cause (exceeded :subject "disk usage" :value "100%") :tense past)'
        )
        atom = atoms[0]
        assert atom.verb == "crashed"
        assert len(atom.contexts) == 1
        assert atom.contexts[0][0] == "cause"
        nested = atom.contexts[0][1]
        assert isinstance(nested, Atom)
        assert nested.verb == "exceeded"

    def test_deep_nesting(self):
        atoms = parse_penman("(believes :subject Alice :object (causes :subject (is :subject X :value Y) :object Z))")
        outer = atoms[0]
        mid = outer.roles[1][1]
        assert isinstance(mid, Atom)
        assert mid.verb == "causes"
        inner = mid.roles[0][1]
        assert isinstance(inner, Atom)
        assert inner.verb == "is"

    def test_multiple_top_level(self):
        atoms = parse_penman(
            "(deployed :subject Alice :object API :tense past) (reviewed :subject Bob :object API :tense past)"
        )
        assert len(atoms) == 2
        assert atoms[0].verb == "deployed"
        assert atoms[1].verb == "reviewed"

    def test_repeated_role(self):
        atoms = parse_penman("(works_with :subject Alice :subject Bob :object project)")
        atom = atoms[0]
        subjects = [(name, val) for name, val in atom.roles if name == "subject"]
        assert len(subjects) == 2
        assert subjects[0][1] == "Alice"
        assert subjects[1][1] == "Bob"

    def test_quoted_strings(self):
        atoms = parse_penman('(has :subject "quick sort" :attribute "time complexity" :value "O(n log n)")')
        atom = atoms[0]
        assert atom.roles[0] == ("subject", "quick sort")
        assert atom.roles[1] == ("attribute", "time complexity")
        assert atom.roles[2] == ("value", "O(n log n)")

    def test_type_coercion_importance_float(self):
        atoms = parse_penman("(test :subject A :object B :importance 0.75)")
        assert atoms[0].modifiers["importance"] == pytest.approx(0.75)

    def test_type_coercion_negated_bool(self):
        atoms = parse_penman("(test :subject A :object B :negated true)")
        assert atoms[0].modifiers["negated"] is True
        atoms2 = parse_penman("(test :subject A :object B :negated false)")
        assert atoms2[0].modifiers["negated"] is False

    def test_negated_yes(self):
        atoms = parse_penman("(test :subject A :object B :negated yes)")
        assert atoms[0].modifiers["negated"] is True

    def test_escaped_quotes_in_string(self):
        atoms = parse_penman(r'(says :subject Alice :object "she said \"hello\"")')
        assert atoms[0].roles[1][1] == 'she said "hello"'

    def test_escape_newline(self):
        atoms = parse_penman(r'(has :subject A :value "line1\nline2")')
        assert atoms[0].roles[1][1] == "line1\nline2"

    def test_escape_tab(self):
        atoms = parse_penman(r'(has :subject A :value "col1\tcol2")')
        assert atoms[0].roles[1][1] == "col1\tcol2"

    def test_escape_backslash(self):
        atoms = parse_penman(r'(has :subject A :value "path\\to\\file")')
        assert atoms[0].roles[1][1] == "path\\to\\file"


# ==================================================================
# TestParseErrors
# ==================================================================


class TestParseErrors:
    def test_unclosed_paren(self):
        with pytest.raises(PenmanParseError, match="Expected RPAREN"):
            parse_penman("(prefers :subject Alice :object Python")

    def test_missing_verb(self):
        with pytest.raises(PenmanParseError, match="Expected verb"):
            parse_penman("(:subject Alice :object Python)")

    def test_empty(self):
        with pytest.raises(PenmanParseError, match="Empty input"):
            parse_penman("")

    def test_whitespace_only(self):
        with pytest.raises(PenmanParseError, match="Empty input"):
            parse_penman("   ")

    def test_error_has_position(self):
        with pytest.raises(PenmanParseError) as exc_info:
            parse_penman("(prefers :subject Alice :object")
        assert exc_info.value.position >= 0

    def test_unterminated_string(self):
        with pytest.raises(PenmanParseError, match="Unterminated"):
            parse_penman('(test :subject "unclosed)')


# ==================================================================
# TestAtomToSentence
# ==================================================================


class TestAtomToSentence:
    def test_subject_verb_object(self):
        atom = Atom(verb="prefers", roles=[("subject", "Alice"), ("object", "Python")])
        s = atom_to_sentence(atom)
        assert s == "Alice prefers Python"

    def test_attribute_value_with_has(self):
        atom = Atom(
            verb="has",
            roles=[("subject", "quick sort"), ("attribute", "time complexity"), ("value", "O(n log n)")],
        )
        s = atom_to_sentence(atom)
        assert s == "quick sort has time complexity O(n log n)"

    def test_attribute_value_with_is(self):
        atom = Atom(
            verb="is",
            roles=[("subject", "Python"), ("attribute", "type"), ("value", "programming language")],
        )
        s = atom_to_sentence(atom)
        assert s == "the type of Python is programming language"

    def test_nested_cause(self):
        inner = Atom(verb="exceeded", roles=[("subject", "disk usage"), ("value", "100%")])
        atom = Atom(
            verb="crashed",
            roles=[("subject", "server")],
            contexts=[("cause", inner)],
        )
        s = atom_to_sentence(atom)
        assert "server" in s
        assert "crashed" in s
        assert "because" in s
        assert "disk usage" in s

    def test_multiple_subjects(self):
        atom = Atom(
            verb="works_with",
            roles=[("subject", "Alice"), ("subject", "Bob"), ("object", "project")],
        )
        s = atom_to_sentence(atom)
        assert "Alice and Bob" in s
        assert "works_with" in s

    def test_all_connectors(self):
        atom = Atom(
            verb="assigned",
            roles=[
                ("subject", "Alice"),
                ("object", "task"),
                ("recipient", "Bob"),
                ("instrument", "Jira"),
                ("origin", "backlog"),
                ("locus", "Monday"),
            ],
        )
        s = atom_to_sentence(atom)
        assert "to Bob" in s
        assert "with Jira" in s
        assert "from backlog" in s
        assert "at Monday" in s

    def test_purpose_connector(self):
        atom = Atom(
            verb="scale",
            roles=[("object", "instances")],
            contexts=[("purpose", "handle load")],
        )
        s = atom_to_sentence(atom)
        assert "in order to handle load" in s

    def test_condition_connector(self):
        inner = Atom(verb="exceeds", roles=[("subject", "CPU"), ("value", "80%")])
        atom = Atom(
            verb="scale",
            roles=[("object", "instances")],
            contexts=[("condition", inner)],
        )
        s = atom_to_sentence(atom)
        assert "if" in s
        assert "CPU" in s

    def test_no_subject(self):
        atom = Atom(verb="deployed", roles=[("object", "API"), ("locus", "prod")])
        s = atom_to_sentence(atom)
        assert "deployed" in s
        assert "API" in s
