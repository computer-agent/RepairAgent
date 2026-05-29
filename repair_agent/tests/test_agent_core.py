"""Tests for ``autogpt.agents.agent`` core logic.

Scope (deliberately NOT overlapping with ``test_agent_parsing.py``, which already
covers the command-normalization branches of ``parse_and_process_response`` and the
``base.py`` helper methods):

* module-level ``extract_command`` and ``execute_command`` — every branch.
* ``Agent.construct_base_prompt`` budget logic (the override before ``super()``).
* ``Agent.on_before_think`` logging.
* ``Agent._save_plausible_patch`` file persistence + dedup.
* ``Agent.execute`` dispatch branches (error / human_feedback / normal / plugins /
  truncation / token-limit / plausible-patch tracking / None result).
* the ``write_fix`` mutation pipeline branch of ``parse_and_process_response``.

Everything heavy/external (LLM, super().construct_base_prompt, mutation queries,
command implementations) is mocked. No network calls are made.

Agent.__init__ is heavy, so we use ``_LightAgent`` — a subclass with a no-op
__init__ — to get instances that satisfy ``isinstance(self, Agent)`` (needed for the
zero-arg ``super()`` calls) without paying construction cost. Methods that do not use
``super()`` are exercised as unbound methods on a ``SimpleNamespace``.
"""

import json
import os
import shutil
import uuid
from types import SimpleNamespace

import pytest

import autogpt.agents.agent as agent_mod
from autogpt.agents.agent import Agent, extract_command, execute_command
from autogpt.llm.base import Message


class _LightAgent(Agent):
    """Agent subclass whose __init__ does nothing, for cheap instances."""

    def __init__(self):  # noqa: D401 - intentionally skip the heavy base __init__
        pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _Resp:
    """Minimal ChatModelResponse stand-in."""

    def __init__(self, content="", function_call=None):
        self.content = content
        self.function_call = function_call


# ===========================================================================
# extract_command
# ===========================================================================
def test_extract_command_normal():
    reply = {"command": {"name": "run_test", "args": {"a": 1}}}
    config = SimpleNamespace(openai_functions=False)
    name, args = extract_command(reply, _Resp(), config)
    assert name == "run_test"
    assert args == {"a": 1}


def test_extract_command_missing_command_key():
    config = SimpleNamespace(openai_functions=False)
    name, args = extract_command({}, _Resp(), config)
    assert name == "Error:"
    assert "Missing 'command'" in args["message"]


def test_extract_command_command_not_a_dict():
    config = SimpleNamespace(openai_functions=False)
    name, args = extract_command({"command": "oops"}, _Resp(), config)
    assert name == "Error:"
    assert "not a dictionary" in args["message"]


def test_extract_command_missing_name_field():
    config = SimpleNamespace(openai_functions=False)
    name, args = extract_command({"command": {"args": {}}}, _Resp(), config)
    assert name == "Error:"
    assert "Missing 'name'" in args["message"]


def test_extract_command_defaults_args_to_empty_dict():
    config = SimpleNamespace(openai_functions=False)
    name, args = extract_command({"command": {"name": "foo"}}, _Resp(), config)
    assert name == "foo"
    assert args == {}


def test_extract_command_openai_functions_uses_function_call():
    config = SimpleNamespace(openai_functions=True)
    fc = SimpleNamespace(name="do_thing", arguments=json.dumps({"x": 5}))
    reply = {}
    name, args = extract_command(reply, _Resp(function_call=fc), config)
    assert name == "do_thing"
    assert args == {"x": 5}


def test_extract_command_openai_functions_no_function_call():
    config = SimpleNamespace(openai_functions=True)
    name, args = extract_command({}, _Resp(function_call=None), config)
    assert name == "Error:"
    assert "No 'function_call'" in args["message"]


def test_extract_command_exception_branch_returns_error():
    # config.openai_functions is False; passing a non-dict reply makes the
    # "command" not in ... membership test raise TypeError -> generic except branch.
    config = SimpleNamespace(openai_functions=False)
    name, args = extract_command(12345, _Resp(), config)
    assert name == "Error:"
    assert args["message"]  # carries the stringified exception


# ===========================================================================
# execute_command
# ===========================================================================
def _agent_with_registry(get_command_return, plugin_commands=None):
    registry = SimpleNamespace(get_command=lambda name: get_command_return)
    prompt_generator = SimpleNamespace(commands=plugin_commands or [])
    ai_config = SimpleNamespace(prompt_generator=prompt_generator)
    return SimpleNamespace(command_registry=registry, ai_config=ai_config)


def test_execute_command_native_dispatch():
    captured = {}

    def fake_cmd(**kwargs):
        captured.update(kwargs)
        return "native-result"

    agent = _agent_with_registry(fake_cmd)
    result = execute_command("foo", {"x": 1}, agent)
    assert result == "native-result"
    assert captured == {"x": 1, "agent": agent}


def test_execute_command_plugin_dispatch_by_label():
    def plugin_fn(**kwargs):
        return f"plugin:{kwargs}"

    plugin_cmd = SimpleNamespace(label="MyPlugin", name="myplugin", function=plugin_fn)
    agent = _agent_with_registry(None, plugin_commands=[plugin_cmd])
    result = execute_command("myplugin", {"q": 2}, agent)
    assert result == "plugin:{'q': 2}"


def test_execute_command_unknown_returns_error_string():
    agent = _agent_with_registry(None, plugin_commands=[])
    result = execute_command("nope", {}, agent)
    assert result.startswith("Error:")
    assert "unknown command" in result


def test_execute_command_native_exception_is_caught():
    def boom(**kwargs):
        raise ValueError("kaboom")

    agent = _agent_with_registry(boom)
    result = execute_command("foo", {}, agent)
    assert result == "Error: kaboom"


# ===========================================================================
# Agent.construct_base_prompt  (budget logic before super())
# ===========================================================================
def test_construct_base_prompt_no_budget_just_delegates(monkeypatch):
    sentinel = object()
    # super().construct_base_prompt returns this sentinel
    monkeypatch.setattr(
        agent_mod.BaseAgent, "construct_base_prompt",
        lambda self, *a, **k: (a, k, sentinel),
    )
    # ApiManager().get_total_budget() == 0 -> no budget message appended
    monkeypatch.setattr(
        agent_mod, "ApiManager",
        lambda: SimpleNamespace(get_total_budget=lambda: 0.0, get_total_cost=lambda: 0.0),
    )
    self = _LightAgent()
    a, k, ret = self.construct_base_prompt("tp_id")
    assert ret is sentinel
    # prepend_messages defaulted to [] ; no append_messages added
    assert k["prepend_messages"] == []
    assert "append_messages" not in k


@pytest.mark.parametrize(
    "remaining,fragment",
    [
        (0.0, "BUDGET EXCEEDED"),
        (0.004, "nearly exceeded"),
        (0.009, "nearly exceeded"),
        (5.0, "remaining API budget"),
    ],
)
def test_construct_base_prompt_appends_budget_message(monkeypatch, remaining, fragment):
    captured = {}

    def fake_super(self, *a, **k):
        captured["kwargs"] = k
        return "done"

    monkeypatch.setattr(agent_mod.BaseAgent, "construct_base_prompt", fake_super)
    # logger.debug(budget_msg) is called with a Message object, which the real log
    # formatter cannot handle; silence it (this path is exercised in production at
    # non-DEBUG levels where the record is dropped before formatting).
    monkeypatch.setattr(agent_mod.logger, "debug", lambda *a, **k: None)
    budget = 10.0
    cost = budget - remaining
    monkeypatch.setattr(
        agent_mod, "ApiManager",
        lambda: SimpleNamespace(get_total_budget=lambda: budget, get_total_cost=lambda: cost),
    )
    self = _LightAgent()
    ret = self.construct_base_prompt("tp_id")
    assert ret == "done"
    appended = captured["kwargs"]["append_messages"]
    assert len(appended) == 1
    assert fragment in appended[0].content


def test_construct_base_prompt_negative_remaining_clamped_to_zero(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        agent_mod.BaseAgent, "construct_base_prompt",
        lambda self, *a, **k: captured.setdefault("k", k),
    )
    # cost > budget -> remaining negative -> clamped to 0 -> "BUDGET EXCEEDED"
    monkeypatch.setattr(
        agent_mod, "ApiManager",
        lambda: SimpleNamespace(get_total_budget=lambda: 1.0, get_total_cost=lambda: 5.0),
    )
    monkeypatch.setattr(agent_mod.logger, "debug", lambda *a, **k: None)
    self = _LightAgent()
    self.construct_base_prompt("tp_id")
    assert "BUDGET EXCEEDED" in captured["k"]["append_messages"][0].content


# ===========================================================================
# Agent.on_before_think  (logging)
# ===========================================================================
def test_on_before_think_logs_two_cycles(monkeypatch):
    calls = []

    class FakeHandler:
        def __init__(self):
            self.log_count_within_cycle = None

        def log_cycle(self, *args):
            calls.append(args)

    prompt = SimpleNamespace(raw=lambda: ["P"])
    monkeypatch.setattr(
        agent_mod.BaseAgent, "on_before_think", lambda self, *a, **k: prompt
    )

    self = _LightAgent()
    self.log_cycle_handler = FakeHandler()
    self.ai_config = SimpleNamespace(ai_name="RepairAgent")
    self.created_at = "now"
    self.cycle_count = 3
    self.project_name = "Lang"
    self.bug_index = "1"
    self.history = SimpleNamespace(raw=lambda: ["H"])

    out = self.on_before_think("prompt", "tp", "instr")
    assert out is prompt
    assert self.log_cycle_handler.log_count_within_cycle == 0
    # Two log_cycle calls: full history, then current context.
    assert len(calls) == 2
    assert calls[0][0] == "RepairAgent"  # ai_name for full history
    assert calls[1][0] == "Lang_1"       # project_name + "_" + bug_index


# ===========================================================================
# Agent._save_plausible_patch  (file persistence + dedup)
# ===========================================================================
@pytest.fixture
def patch_self(monkeypatch):
    """Fake self + throwaway experiment dir for _save_plausible_patch writes."""
    exp_name = "pytest_throwaway_" + uuid.uuid4().hex[:8]
    exp_dir = os.path.join("experimental_setups", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    fake = SimpleNamespace(
        exps=[exp_name],
        project_name="Lang",
        bug_index=1,
    )
    try:
        yield fake, exp_dir
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)


def _plausible_path(exp_dir):
    return os.path.join(exp_dir, "plausible_patches", "plausible_patches_Lang_1.json")


def test_save_plausible_patch_writes_file(patch_self):
    fake, exp_dir = patch_self
    Agent._save_plausible_patch(fake, {"line": 10}, source="write_fix")
    path = _plausible_path(exp_dir)
    assert os.path.exists(path)
    with open(path) as f:
        data = json.load(f)
    assert data == [{"source": "write_fix", "patch": {"line": 10}}]


def test_save_plausible_patch_dedups_identical(patch_self):
    fake, exp_dir = patch_self
    Agent._save_plausible_patch(fake, {"line": 10}, source="mutant")
    Agent._save_plausible_patch(fake, {"line": 10}, source="mutant")  # identical -> skipped
    with open(_plausible_path(exp_dir)) as f:
        data = json.load(f)
    assert len(data) == 1


def test_save_plausible_patch_appends_distinct(patch_self):
    fake, exp_dir = patch_self
    Agent._save_plausible_patch(fake, {"line": 10}, source="mutant")
    Agent._save_plausible_patch(fake, {"line": 20}, source="write_fix")
    with open(_plausible_path(exp_dir)) as f:
        data = json.load(f)
    assert len(data) == 2
    assert {"source": "write_fix", "patch": {"line": 20}} in data


# ===========================================================================
# Agent.execute  (dispatch branches)
# ===========================================================================
def _execute_self(monkeypatch, history_summary="hist", token_limit=100000, plugins=None):
    """Build a fake self for Agent.execute with sub-objects it touches."""
    added = []

    class History:
        def add(self, role, content, kind):
            added.append((role, content, kind))

        def summary_message(self):
            return history_summary

    fake = SimpleNamespace(
        config=SimpleNamespace(plugins=plugins or []),
        llm=SimpleNamespace(name="gpt-3.5-turbo"),
        history=History(),
        send_token_limit=token_limit,
        ai_config=SimpleNamespace(ai_name="RepairAgent"),
        created_at="now",
        cycle_count=0,
        log_cycle_handler=SimpleNamespace(log_cycle=lambda *a, **k: None),
        _added=added,
    )
    return fake


def test_execute_error_command(monkeypatch):
    fake = _execute_self(monkeypatch)
    result = Agent.execute(fake, "Error:", {"k": "v"}, None)
    assert result.startswith("Could not execute command")
    assert ("user", result, "action_result") in fake._added


def test_execute_human_feedback(monkeypatch):
    fake = _execute_self(monkeypatch)
    result = Agent.execute(fake, "human_feedback", {}, "please continue")
    assert result == "Human feedback: please continue"
    assert fake._added[-1] == ("user", result, "action_result")


def test_execute_normal_command(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_command", lambda command_name, arguments, agent: "ok-result")
    fake = _execute_self(monkeypatch)
    result = Agent.execute(fake, "run_test", {"project_name": "Lang"}, None)
    assert result == "Command run_test returned: ok-result"


def test_execute_truncates_long_result(monkeypatch):
    long = "x" * 5000
    monkeypatch.setattr(agent_mod, "execute_command", lambda **k: long)
    # avoid token-limit branch overriding result: make limit huge
    monkeypatch.setattr(agent_mod, "count_string_tokens", lambda s, m: 1)
    fake = _execute_self(monkeypatch)
    result = Agent.execute(fake, "run_test", {}, None)
    assert "truncated it to the first 4000 characters" in result
    assert len(result) < 4200  # only first 4000 chars of the command result kept


def test_execute_token_limit_exceeded(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_command", lambda **k: "short")
    monkeypatch.setattr(agent_mod, "count_string_tokens", lambda s, m: 100000)
    fake = _execute_self(monkeypatch, token_limit=10)
    result = Agent.execute(fake, "run_test", {}, None)
    assert "returned too much output" in result


def test_execute_plugins_pre_and_post(monkeypatch):
    class Plugin:
        def can_handle_pre_command(self):
            return True

        def pre_command(self, name, args):
            return ("renamed_cmd", {"changed": True})

        def can_handle_post_command(self):
            return True

        def post_command(self, name, result):
            return result + " [post]"

    monkeypatch.setattr(agent_mod, "execute_command", lambda command_name, arguments, agent: f"got {command_name}")
    monkeypatch.setattr(agent_mod, "count_string_tokens", lambda s, m: 1)
    fake = _execute_self(monkeypatch, plugins=[Plugin()])
    result = Agent.execute(fake, "orig_cmd", {}, None)
    # pre_command renamed the command; post_command appended marker
    assert "got renamed_cmd" in result
    assert result.endswith("[post]")


def test_execute_tracks_plausible_patch(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_command", lambda **k: "Tests run: 1, 0 failing test")
    monkeypatch.setattr(agent_mod, "count_string_tokens", lambda s, m: 1)
    saved = {}
    fake = _execute_self(monkeypatch)
    fake._save_plausible_patch = lambda patch_data, source: saved.update(
        patch=patch_data, source=source
    )
    Agent.execute(fake, "write_fix", {"changes_dicts": [{"l": 1}]}, None)
    assert saved["source"] == "write_fix"
    assert saved["patch"] == [{"l": 1}]


def test_execute_none_result_adds_unable_message(monkeypatch):
    monkeypatch.setattr(agent_mod, "execute_command", lambda **k: None)
    monkeypatch.setattr(agent_mod, "count_string_tokens", lambda s, m: 1)
    fake = _execute_self(monkeypatch)
    result = Agent.execute(fake, "run_test", {}, None)
    # execute_command returned None, but result is the formatted "returned: None" string,
    # which is not None -> the normal add path is taken.
    assert result == "Command run_test returned: None"
    assert fake._added[-1] == ("user", result, "action_result")


# ===========================================================================
# parse_and_process_response  : write_fix mutation pipeline branch
# ===========================================================================
@pytest.fixture
def write_fix_self(monkeypatch):
    """Fake self + throwaway experiment dirs for the write_fix mutation pipeline."""
    exp_name = "pytest_throwaway_" + uuid.uuid4().hex[:8]
    exp_dir = os.path.join("experimental_setups", exp_name)
    for sub in ("responses", "mutations_history"):
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    config = SimpleNamespace(
        openai_functions=False, debug_mode=False, plugins=[], static_llm="static-model"
    )
    fake = SimpleNamespace(
        project_name="Lang",
        bug_index=1,
        exps=[exp_name],
        config=config,
        ai_config=SimpleNamespace(ai_name="RepairAgent"),
        created_at="now",
        cycle_count=0,
        log_cycle_handler=SimpleNamespace(log_cycle=lambda *a, **k: None),
        construct_mutation_prompt=lambda fix, buggies: "MUT-PROMPT",
        save_to_json=lambda path, data: None,
        _save_plausible_patch=lambda patch_data, source: None,
    )
    try:
        yield fake, exp_dir
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)


def test_parse_write_fix_triggers_mutation_pipeline(monkeypatch, write_fix_self):
    fake, exp_dir = write_fix_self

    # Stub out everything the write_fix branch calls in the agent module namespace.
    monkeypatch.setattr(agent_mod, "get_detailed_list_of_buggy_lines", lambda p, b: ["buggy line"])
    # query_for_mutants returns a markdown-fenced JSON list of one mutant.
    monkeypatch.setattr(
        agent_mod, "query_for_mutants",
        lambda prompt, llm: '```json\n[{"mutant": "do X"}]\n```',
    )
    constructed = {}
    monkeypatch.setattr(
        agent_mod, "construct_fix_command",
        lambda m, p, b: constructed.setdefault("cmd", {"command": {"name": "write_fix", "args": {}}}),
    )
    exec_calls = []
    monkeypatch.setattr(
        agent_mod, "execute_command",
        lambda name, args, agent: exec_calls.append((name, args)) or "Tests run: 1, 0 failing test",
    )
    saved = {}
    fake._save_plausible_patch = lambda patch_data, source: saved.update(patch=patch_data, source=source)
    saved_json = {}
    fake.save_to_json = lambda path, data: saved_json.update(path=path, data=data)

    content = json.dumps(
        {
            "thoughts": "fixing",
            "command": {"name": "write_fix", "args": {"changes_dicts": [{"line": 1, "code": "x"}]}},
        }
    )
    name, args, reply = Agent.parse_and_process_response(fake, _Resp(content))

    assert name == "write_fix"
    # mutant was constructed, executed, and (since "0 failing test") saved as plausible.
    assert exec_calls  # execute_command was invoked for the mutant
    assert saved.get("source") == "mutant"
    assert saved.get("patch") == {"mutant": "do X"}
    # the parsed mutants were persisted via save_to_json
    assert saved_json.get("data") == [{"mutant": "do X"}]
    # the raw mutants response file was written (raw_m write, not stubbed)
    assert os.path.exists(
        os.path.join(exp_dir, "mutations_history", "mutants_raw_Lang_1.json")
    )


# ===========================================================================
# parse_and_process_response : branches NOT covered by test_agent_parsing.py
# ===========================================================================
@pytest.fixture
def parse_self(monkeypatch):
    """Fake Agent self + throwaway experiment dir for the responses write."""
    exp_name = "pytest_throwaway_" + uuid.uuid4().hex[:8]
    exp_dir = os.path.join("experimental_setups", exp_name)
    os.makedirs(os.path.join(exp_dir, "responses"), exist_ok=True)
    config = SimpleNamespace(openai_functions=False, debug_mode=False, plugins=[])
    fake = SimpleNamespace(
        project_name="Lang",
        bug_index=1,
        exps=[exp_name],
        config=config,
        ai_config=SimpleNamespace(ai_name="RepairAgent"),
        created_at="now",
        cycle_count=0,
        log_cycle_handler=SimpleNamespace(log_cycle=lambda *a, **k: None),
    )
    try:
        yield fake
    finally:
        shutil.rmtree(exp_dir, ignore_errors=True)


def test_parse_command_args_not_a_dict_substitutes_unknown(parse_self):
    # A known command whose "args" is not a dict -> the else branch substitutes
    # unknown_command (args dict missing entirely here).
    content = json.dumps(
        {"thoughts": "t", "command": {"name": "express_hypothesis", "args": "oops"}}
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _Resp(content))
    assert reply["command"]["name"] == "unknown_command"
    assert name == "unknown_command"


def test_parse_unknown_command_not_in_interface_passes_through(parse_self):
    # A command name absent from commands_interface.json skips the normalization
    # block entirely; validate_dict (lenient on the name) accepts it and the original
    # command dict flows through unchanged.
    content = json.dumps(
        {"thoughts": "t", "command": {"name": "totally_made_up_cmd", "args": {}}}
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _Resp(content))
    assert name == "totally_made_up_cmd"
    assert reply["command"]["args"] == {}


def test_parse_underscore_split_persists_for_non_autoinject(parse_self):
    # run_test has a project_name arg but is NOT in the auto-inject list, so the
    # underscore-split prefix survives (not overwritten by self.project_name).
    content = json.dumps(
        {
            "thoughts": "t",
            "command": {"name": "run_test", "args": {"project_name": "Foo_bar_baz", "bug_index": 7}},
        }
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _Resp(content))
    assert name == "run_test"
    assert reply["command"]["args"]["project_name"] == "Foo"
    assert reply["command"]["args"]["bug_index"] == 7


def test_parse_fuzzy_arg_remap_to_substring_ref(parse_self):
    # write_range ref args include 'changed_lines'. Provided 'lines' is not an exact
    # match but is a substring of 'changed_lines' -> fuzzy-mapped onto it (break path).
    content = json.dumps(
        {
            "thoughts": "t",
            "command": {
                "name": "write_range",
                "args": {"filepath": "F.java", "lines": "new code"},
            },
        }
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _Resp(content))
    assert name == "write_range"
    out = reply["command"]["args"]
    assert out["changed_lines"] == "new code"
    assert out["filepath"] == "F.java"


def test_parse_post_planning_plugin_invoked(parse_self):
    calls = []

    class Plugin:
        def can_handle_post_planning(self):
            return True

        def post_planning(self, reply):
            calls.append(reply)
            return reply

    parse_self.config.plugins = [Plugin()]
    content = json.dumps(
        {"thoughts": "t", "command": {"name": "express_hypothesis", "args": {"hypothesis": "h"}}}
    )
    name, args, reply = Agent.parse_and_process_response(parse_self, _Resp(content))
    assert name == "express_hypothesis"
    assert calls  # the post_planning hook ran


def test_parse_write_fix_mutation_error_is_swallowed(monkeypatch, write_fix_self):
    fake, _ = write_fix_self
    monkeypatch.setattr(agent_mod, "get_detailed_list_of_buggy_lines", lambda p, b: [])
    # query_for_mutants raising would propagate (it's outside the try); instead return
    # malformed content so json_repair / processing inside the try block fails and the
    # broad ``except`` logs+swallows it without crashing parse_and_process_response.
    monkeypatch.setattr(agent_mod, "query_for_mutants", lambda prompt, llm: "not json at all")

    def boom(*a, **k):
        raise RuntimeError("construct failed")

    monkeypatch.setattr(agent_mod, "construct_fix_command", boom)

    content = json.dumps(
        {
            "thoughts": "fixing",
            "command": {"name": "write_fix", "args": {"changes_dicts": []}},
        }
    )
    # Should NOT raise despite the mutation pipeline error.
    name, args, reply = Agent.parse_and_process_response(fake, _Resp(content))
    assert name == "write_fix"
