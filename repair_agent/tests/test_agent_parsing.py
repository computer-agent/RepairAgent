"""Tests for agent command handling / parsing.

Two areas are covered:

(A) Standalone-ish methods of ``autogpt.agents.base.BaseAgent`` exercised as UNBOUND
    methods against a hand-built ``SimpleNamespace`` ``self`` that only carries the
    attributes each method actually touches. This avoids constructing a real (very
    heavy) agent.

(B) ``autogpt.agents.agent.Agent.parse_and_process_response`` command-normalization
    logic, again called as an unbound method on a fake ``self``. A throwaway experiment
    directory is created under ``experimental_setups`` for the model_responses write and
    cleaned up afterwards. No network / LLM calls are made.
"""

import json
import os
import shutil
import uuid
from types import SimpleNamespace

import pytest

from autogpt.agents.agent import Agent
from autogpt.agents.base import BaseAgent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Msg:
    """Minimal stand-in for a history message (has .role and .content)."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


def assistant_cmd(name, args=None, thoughts="t"):
    """Build assistant message content carrying a JSON command dict."""
    return json.dumps(
        {"thoughts": thoughts, "command": {"name": name, "args": args or {}}}
    )


# ===========================================================================
# (A) base.py : detect_command_repetition
# ===========================================================================
def test_detect_command_repetition_true_when_repeated():
    ref = {
        "command": {
            "name": "run_test",
            "args": {"project_name": "Lang", "bug_index": 1},
        }
    }
    history = [
        Msg("user", "do something"),
        Msg(
            "assistant",
            assistant_cmd("run_test", {"project_name": "Lang", "bug_index": 1}),
        ),
    ]
    fake_self = SimpleNamespace(history=history)
    assert BaseAgent.detect_command_repetition(fake_self, ref) is True


def test_detect_command_repetition_false_when_different():
    ref = {
        "command": {
            "name": "run_test",
            "args": {"project_name": "Lang", "bug_index": 1},
        }
    }
    history = [
        Msg("assistant", assistant_cmd("read_range", {"filepath": "a.java"})),
    ]
    fake_self = SimpleNamespace(history=history)
    assert BaseAgent.detect_command_repetition(fake_self, ref) is False


def test_detect_command_repetition_only_considers_assistant_messages():
    # The same command text appears, but in a 'user' message -> must be ignored.
    ref = {"command": {"name": "run_test", "args": {}}}
    history = [Msg("user", assistant_cmd("run_test", {}))]
    fake_self = SimpleNamespace(history=history)
    assert BaseAgent.detect_command_repetition(fake_self, ref) is False


def test_detect_command_repetition_returns_false_on_bad_ref():
    # ref_cmd missing the 'command' key triggers the except branch -> False.
    history = [Msg("assistant", assistant_cmd("run_test", {}))]
    fake_self = SimpleNamespace(history=history)
    assert BaseAgent.detect_command_repetition(fake_self, {}) is False


# ===========================================================================
# (A) base.py : handle_command_repetition
# ===========================================================================
def test_handle_command_repetition_empty_strategy_returns_empty():
    assert (
        BaseAgent.handle_command_repetition(SimpleNamespace(), {"command": "x"}, "")
        == ""
    )


def test_handle_command_repetition_restrict_mentions_command():
    out = BaseAgent.handle_command_repetition(
        SimpleNamespace(), {"command": "run_test"}, "RESTRICT"
    )
    assert "totally different" in out
    assert "run_test" in out


def test_handle_command_repetition_top3_does_not_need_command():
    out = BaseAgent.handle_command_repetition(
        SimpleNamespace(), {"command": "x"}, "TOP3"
    )
    assert "three commands" in out


def test_handle_command_repetition_unsupported_strategy_raises():
    with pytest.raises(ValueError):
        BaseAgent.handle_command_repetition(SimpleNamespace(), {"command": "x"}, "NOPE")


# ===========================================================================
# (A) base.py : update_prompt_state + switch_state transitions
# ===========================================================================
def _switch_self(history):
    """Build a fake self for switch_state with the dicts update_prompt_state touches."""
    return SimpleNamespace(
        history=history,
        current_state="collect information to understand the bug",
        descriptions={
            "collect information to understand the bug": "DESC-UNDERSTAND",
            "collect information to fix the bug": "DESC-FIX",
            "trying out candidate fixes": "DESC-TRYING",
        },
        cmds_by_state={
            "collect information to understand the bug": "CMDS-UNDERSTAND",
            "collect information to fix the bug": "CMDS-FIX",
            "trying out candidate fixes": "CMDS-TRYING",
        },
        prompt_dictionary={"current state": "", "commands": [0, 1, "OLD"]},
    )


def _bind_update_prompt_state(fake):
    """switch_state calls self.update_prompt_state; wire it to the real method."""
    fake.update_prompt_state = lambda state: BaseAgent.update_prompt_state(fake, state)
    return fake


def test_update_prompt_state_sets_all_three():
    fake = _switch_self([])
    BaseAgent.update_prompt_state(fake, "trying out candidate fixes")
    assert fake.current_state == "trying out candidate fixes"
    assert fake.prompt_dictionary["current state"] == "DESC-TRYING"
    assert fake.prompt_dictionary["commands"][2] == "CMDS-TRYING"


def test_switch_state_transition_to_fix_via_hypothesis():
    # switch_state scans from the end; index i must be 'assistant' and it reads
    # history[i+1].content. Build [assistant, user-with-trigger].
    trigger = (
        "Since you have a hypothesis about the bug, the current state have been "
        "changed from 'collect information to understand the bug' to 'collect "
        "information to fix the bug'"
    )
    # switch_state's loop is range(len-1, 0, -1): it never inspects index 0, so the
    # assistant message must sit at index >= 1 (with its result at i+1).
    history = [Msg("user", "filler"), Msg("assistant", "cmd"), Msg("user", trigger)]
    fake = _bind_update_prompt_state(_switch_self(history))
    BaseAgent.switch_state(fake)
    assert fake.current_state == "collect information to fix the bug"
    assert fake.prompt_dictionary["current state"] == "DESC-FIX"


def test_switch_state_transition_to_trying_candidate_fixes():
    trigger = "\n **Note:** You are automatically switched to the state 'trying out candidate fixes'"
    history = [Msg("user", "filler"), Msg("assistant", "cmd"), Msg("user", trigger)]
    fake = _bind_update_prompt_state(_switch_self(history))
    BaseAgent.switch_state(fake)
    assert fake.current_state == "trying out candidate fixes"


def test_switch_state_no_transition_when_no_trigger():
    history = [
        Msg("user", "filler"),
        Msg("assistant", "cmd"),
        Msg("user", "nothing special here"),
    ]
    fake = _bind_update_prompt_state(_switch_self(history))
    BaseAgent.switch_state(fake)
    # unchanged
    assert fake.current_state == "collect information to understand the bug"


# ===========================================================================
# (A) base.py : validate_command_parsing (used by construct_read_files etc.)
# ===========================================================================
def test_validate_command_parsing_accepts_exact_args():
    # express_hypothesis requires exactly ["hypothesis"]
    cmd = {"command": {"name": "express_hypothesis", "args": {"hypothesis": "h"}}}
    assert BaseAgent.validate_command_parsing(SimpleNamespace(), cmd) is True


def test_validate_command_parsing_rejects_extra_or_missing_args():
    cmd = {
        "command": {
            "name": "express_hypothesis",
            "args": {"hypothesis": "h", "extra": 1},
        }
    }
    assert BaseAgent.validate_command_parsing(SimpleNamespace(), cmd) is False


def test_validate_command_parsing_rejects_unknown_command():
    cmd = {"command": {"name": "not_a_real_command", "args": {}}}
    assert BaseAgent.validate_command_parsing(SimpleNamespace(), cmd) is False


# ===========================================================================
# (A) base.py : construct_read_files
# ===========================================================================
def test_construct_read_files_pairs_command_with_next_result():
    read_cmd = assistant_cmd(
        "read_range",
        {
            "project_name": "Lang",
            "bug_index": 1,
            "filepath": "Foo.java",
            "startline": 10,
            "endline": 20,
        },
    )
    history = [
        Msg("assistant", read_cmd),
        Msg("user", "lines 10-20 content here"),
    ]
    fake = SimpleNamespace(history=history, read_files={})
    # construct_read_files calls self.validate_command_parsing internally.
    fake.validate_command_parsing = lambda cd: BaseAgent.validate_command_parsing(
        fake, cd
    )
    BaseAgent.construct_read_files(fake)
    assert fake.read_files == {"Foo.java": {"10,20": "lines 10-20 content here"}}


def test_construct_read_files_empty_when_no_read_commands():
    history = [
        Msg(
            "assistant",
            assistant_cmd("run_test", {"project_name": "Lang", "bug_index": 1}),
        )
    ]
    fake = SimpleNamespace(history=history, read_files={})
    fake.validate_command_parsing = lambda cd: BaseAgent.validate_command_parsing(
        fake, cd
    )
    BaseAgent.construct_read_files(fake)
    assert fake.read_files == {}


# ===========================================================================
# (B) agent.py : parse_and_process_response command normalization
# ===========================================================================
@pytest.fixture
def parse_self(monkeypatch):
    """Build a fake Agent 'self' + a throwaway experiment dir for the responses write.

    Yields the SimpleNamespace self. validate_dict / extract_command are exercised for
    real (they only need config flags), so the response dict must have exactly the keys
    {thoughts, command} to satisfy the JSON schema (additionalProperties: false).
    """
    exp_name = "pytest_throwaway_" + uuid.uuid4().hex[:8]
    exp_dir = os.path.join("experimental_setups", exp_name)
    os.makedirs(os.path.join(exp_dir, "responses"), exist_ok=True)

    config = SimpleNamespace(openai_functions=False, debug_mode=False, plugins=[])
    log_handler = SimpleNamespace(log_cycle=lambda *a, **k: None)
    fake = SimpleNamespace(
        project_name="Lang",
        bug_index=1,
        exps=[exp_name],
        config=config,
        ai_config=SimpleNamespace(ai_name="RepairAgent"),
        created_at="now",
        cycle_count=0,
        log_cycle_handler=log_handler,
    )
    try:
        yield fake
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)


def _resp(content):
    return SimpleNamespace(content=content)


def test_parse_missing_command_substitutes_missing_command(parse_self):
    # No 'command' key -> name SHOULD become 'missing_command'. Must keep top-level keys
    # to exactly {thoughts, command} so the schema validates.
    content = json.dumps({"thoughts": "just thinking"})
    name, args, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    assert reply["command"]["name"] == "missing_command"
    assert name == "missing_command"


def test_parse_non_dict_command_substitutes_unknown_command(parse_self):
    content = json.dumps({"thoughts": "x", "command": "not a dict"})
    name, args, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    assert reply["command"]["name"] == "unknown_command"
    assert name == "unknown_command"


def test_parse_known_command_keeps_only_known_args_and_injects_ids(parse_self):
    # read_range is in the auto-inject list -> project_name/bug_index are forced to
    # the agent's values; unknown args are dropped.
    content = json.dumps(
        {
            "thoughts": "reading",
            "command": {
                "name": "read_range",
                "args": {
                    "filepath": "Foo.java",
                    "startline": 5,
                    "endline": 9,
                    "garbage_arg": "drop me",
                },
            },
        }
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    out_args = reply["command"]["args"]
    assert name == "read_range"
    assert "garbage_arg" not in out_args
    assert out_args["filepath"] == "Foo.java"
    assert out_args["startline"] == 5
    assert out_args["endline"] == 9
    # auto-injected from self.project_name / self.bug_index
    assert out_args["project_name"] == "Lang"
    assert out_args["bug_index"] == 1


def test_parse_project_name_with_underscore_is_split_to_prefix(parse_self):
    # search_code_base is in the auto-inject list. project_name is overwritten with
    # self.project_name ("Lang") AFTER the underscore-split, so the final value is
    # always self.project_name. We assert the documented behavior: prefix 'Lang'.
    content = json.dumps(
        {
            "thoughts": "searching",
            "command": {
                "name": "search_code_base",
                "args": {"key_words": ["foo"], "project_name": "Lang_extra_suffix"},
            },
        }
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    assert name == "search_code_base"
    assert reply["command"]["args"]["project_name"] == "Lang"
    assert reply["command"]["args"]["key_words"] == ["foo"]


def test_parse_known_command_without_autoinject_keeps_provided_args(parse_self):
    # express_hypothesis is NOT in the auto-inject list; with the underscore split,
    # a project_name is not present so nothing is injected. Only known args survive.
    content = json.dumps(
        {
            "thoughts": "hypothesis",
            "command": {
                "name": "express_hypothesis",
                "args": {"hypothesis": "h1", "junk": 2},
            },
        }
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    assert name == "express_hypothesis"
    assert reply["command"]["args"] == {"hypothesis": "h1"}


def test_parse_unmatched_arg_mapped_to_substring_ref(parse_self):
    # The fuzzy-mapping block maps an unmatched provided arg onto a ref arg if the
    # provided name is a substring of the ref name. 'file' is a substring of
    # 'file_path' (ref for get_classes_and_methods). project_name/bug_index injected.
    content = json.dumps(
        {
            "thoughts": "classes",
            "command": {
                "name": "get_classes_and_methods",
                "args": {"file": "Foo.java"},
            },
        }
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    assert name == "get_classes_and_methods"
    assert reply["command"]["args"]["file_path"] == "Foo.java"
    assert reply["command"]["args"]["project_name"] == "Lang"


def test_parse_empty_content_raises_syntaxerror(parse_self):
    with pytest.raises(SyntaxError):
        Agent.parse_and_process_response(parse_self, _resp(""))


# ===========================================================================
# Bug-hunt regressions: fuzzy arg-remapping must pick the most specific ref and
# must not collapse two distinct provided args onto one ref (silent data loss).
# ===========================================================================
def test_fuzzy_arg_maps_name_to_method_name_not_project_name(parse_self):
    # "name" is a substring of BOTH project_name and method_name; it must bind to
    # method_name (the specific field the model means), not project_name (which is
    # auto-injected and would silently drop the value).
    content = json.dumps(
        {
            "thoughts": "x",
            "command": {
                "name": "extract_method_code",
                "args": {"file": "Foo.java", "name": "doStuff"},
            },
        }
    )
    _, _, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    a = reply["command"]["args"]
    assert a.get("method_name") == "doStuff"
    assert a.get("filepath") == "Foo.java"
    assert a.get("project_name") == "Lang"  # auto-injected, not clobbered by "name"


def test_fuzzy_arg_no_collapse_two_args_onto_one_ref(parse_self):
    # "changed" and "lines" both fuzzy-match changed_lines; the ref must be filled
    # by exactly one (first match wins), never silently last-writer-clobbered.
    content = json.dumps(
        {
            "thoughts": "x",
            "command": {
                "name": "write_range",
                "args": {"filepath": "F.java", "changed": "AAA", "lines": "BBB"},
            },
        }
    )
    _, _, reply = Agent.parse_and_process_response(parse_self, _resp(content))
    a = reply["command"]["args"]
    assert a.get("changed_lines") == "AAA"
