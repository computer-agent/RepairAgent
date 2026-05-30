"""Tests for autogpt/json_utils/utilities.py.

Covers extract_dict_from_response (parsing of various LLM response shapes),
llm_response_schema (loading + openai_functions stripping), and validate_dict
(schema validation against llm_response_format_1.json).

The LLM is never called here: extract_dict_from_response is pure string/JSON
handling, and validate_dict only reads a bundled schema file plus a small Config
stand-in (SimpleNamespace).
"""

from types import SimpleNamespace

import pytest

from autogpt.json_utils.utilities import (
    extract_dict_from_response,
    llm_response_schema,
    validate_dict,
)


def _cfg(openai_functions=False, debug_mode=False):
    """Minimal Config stand-in: validate_dict / llm_response_schema only read
    these two attributes."""
    return SimpleNamespace(openai_functions=openai_functions, debug_mode=debug_mode)


# --------------------------------------------------------------------------- #
# extract_dict_from_response
# --------------------------------------------------------------------------- #
def test_plain_json_object():
    assert extract_dict_from_response('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_json_in_json_code_block():
    resp = '```json\n{"a": 1}\n```'
    assert extract_dict_from_response(resp) == {"a": 1}


def test_json_in_plain_code_block():
    resp = '```\n{"a": 1}\n```'
    assert extract_dict_from_response(resp) == {"a": 1}


def test_python_dict_single_quotes():
    # repair_json / ast fallback normalizes single quotes to a real dict
    assert extract_dict_from_response("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}


def test_trailing_comma_is_repaired():
    assert extract_dict_from_response('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_empty_string_returns_empty_dict():
    assert extract_dict_from_response("") == {}


def test_whitespace_only_returns_empty_dict():
    assert extract_dict_from_response("   \n  \t ") == {}


def test_json_object_embedded_in_prose_code_block():
    resp = 'Here is my answer:\n```json\n{"a": 1, "c": true}\n```\nDone.'
    assert extract_dict_from_response(resp) == {"a": 1, "c": True}


def test_nested_object_preserved():
    resp = '{"thoughts": "hi", "command": {"name": "do", "args": {"x": 1}}}'
    assert extract_dict_from_response(resp) == {
        "thoughts": "hi",
        "command": {"name": "do", "args": {"x": 1}},
    }


# The following two cases (valid JSON that is NOT a dict) fall through to the
# Valid JSON that is NOT an object (array, bare number) must fall through to the
# {} return rather than yielding a non-dict. (Earlier a logger.warning call here
# crashed with AttributeError because the custom Logger only defines `warn`; that
# is fixed, so the documented {} contract holds.)
def test_json_array_returns_empty_dict():
    assert extract_dict_from_response("[1, 2, 3]") == {}


def test_bare_number_returns_empty_dict():
    assert extract_dict_from_response("42") == {}


# --------------------------------------------------------------------------- #
# llm_response_schema
# --------------------------------------------------------------------------- #
def test_schema_has_expected_required_fields():
    schema = llm_response_schema(_cfg(openai_functions=False))
    assert schema["required"] == ["thoughts", "command"]
    assert "command" in schema["properties"]
    assert "thoughts" in schema["properties"]


def test_schema_strips_command_when_openai_functions():
    schema = llm_response_schema(_cfg(openai_functions=True))
    assert "command" not in schema["properties"]
    assert "command" not in schema["required"]
    assert "thoughts" in schema["properties"]


# --------------------------------------------------------------------------- #
# validate_dict
# --------------------------------------------------------------------------- #
def _valid_object():
    return {
        "thoughts": "I will inspect the failing test.",
        "command": {"name": "read_file", "args": {"path": "Foo.java"}},
    }


def test_validate_dict_valid_object():
    ok, errors = validate_dict(_valid_object(), _cfg())
    assert ok is True
    assert errors is None


def test_validate_dict_missing_required_field():
    obj = {"command": {"name": "read_file", "args": {}}}  # missing "thoughts"
    ok, errors = validate_dict(obj, _cfg())
    assert ok is False
    assert isinstance(errors, list)
    assert len(errors) >= 1


def test_validate_dict_command_missing_args():
    obj = {"thoughts": "x", "command": {"name": "read_file"}}  # args required
    ok, errors = validate_dict(obj, _cfg())
    assert ok is False
    assert isinstance(errors, list) and errors


def test_validate_dict_additional_property_rejected():
    obj = _valid_object()
    obj["extra"] = "nope"  # additionalProperties: false
    ok, errors = validate_dict(obj, _cfg())
    assert ok is False
    assert isinstance(errors, list) and errors


def test_validate_dict_wrong_type_for_thoughts():
    obj = {"thoughts": 123, "command": {"name": "do", "args": {}}}
    ok, errors = validate_dict(obj, _cfg())
    assert ok is False
    assert isinstance(errors, list) and errors


def test_validate_dict_debug_mode_still_returns_errors():
    # debug_mode triggers extra logging branches; behavior/return must be unchanged.
    obj = {"command": {"name": "do", "args": {}}}
    ok, errors = validate_dict(obj, _cfg(debug_mode=True))
    assert ok is False
    assert isinstance(errors, list) and errors


# --------------------------------------------------------------------------- #
# extract_dict_from_response — robustness against realistic LLM formatting.
# These reveal bugs in the fragile fence-slicing logic: the whole agent loop
# depends on recovering the {"thoughts":.., "command":..} object, and returning
# {} or a truncated dict causes a no-op cycle (the issue #21 looping symptom).
# --------------------------------------------------------------------------- #
def test_real_json_in_second_code_block_is_recovered():
    # An illustrative block precedes the real answer block.
    resp = (
        "For example you could do:\n```\nsome illustrative non-json text\n```\n"
        "But here is my actual answer:\n```json\n"
        '{"thoughts": "ok", "command": {"name": "read_file", "args": {"path": "Foo.java"}}}\n```'
    )
    assert extract_dict_from_response(resp) == {
        "thoughts": "ok",
        "command": {"name": "read_file", "args": {"path": "Foo.java"}},
    }


def test_triple_backtick_inside_string_value_does_not_truncate():
    resp = '```json\n{"thoughts": "use ```code``` here", "command": {"name": "x", "args": {}}}\n```'
    assert extract_dict_from_response(resp) == {
        "thoughts": "use ```code``` here",
        "command": {"name": "x", "args": {}},
    }


def test_json_on_same_line_as_opening_fence_is_recovered():
    resp = '```json {"thoughts": "ok", "command": {"name": "x", "args": {}}}```'
    assert extract_dict_from_response(resp) == {
        "thoughts": "ok",
        "command": {"name": "x", "args": {}},
    }
