"""Unit tests for ``autogpt.agents.base`` (BaseAgent + module function).

All methods are exercised as UNBOUND functions against a hand-built
``SimpleNamespace`` ``self`` that carries only the attributes each method reads.
Constructing a real ``BaseAgent`` is avoided (its ``__init__`` does heavy file IO,
git checkouts and LLM model setup).

The methods covered here are DISJOINT from ``test_agent_parsing.py`` (which already
covers detect_command_repetition, handle_command_repetition, update_prompt_state,
switch_state, validate_command_parsing, construct_read_files). Here we focus on the
large family of ``construct_*`` history-parsing builders, the ``construct_*_context``
string builders, ``save_to_json``, ``load_context``, ``response_format_instruction``,
``on_before_think`` and the module-level ``add_history_upto_token_limit``.

No network / LLM / subprocess calls are made; file IO is redirected to tmp_path.
"""

import json
import os
from types import SimpleNamespace

import pytest

import autogpt.agents.base as base_mod
from autogpt.agents.base import BaseAgent, add_history_upto_token_limit
from autogpt.llm.base import ChatSequence, Message
from autogpt.memory.message_history import MessageHistory


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class Msg:
    """Minimal stand-in for a history message (has .role and .content)."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


def assistant_cmd(name, args=None, thoughts="reasoning-text"):
    return json.dumps({"thoughts": thoughts, "command": {"name": name, "args": args or {}}})


def bind_validate(fake):
    """Wire self.validate_command_parsing to the real (file-reading) implementation."""
    fake.validate_command_parsing = lambda cd: BaseAgent.validate_command_parsing(fake, cd)
    return fake


# ===========================================================================
# construct_*_context : pure string builders (empty vs populated branches)
# ===========================================================================
def test_construct_hypothesises_context_empty():
    out = BaseAgent.construct_hypothesises_context(SimpleNamespace(hypothesises=[]))
    assert "No hypothesis made yet." in out
    assert out.startswith("## Hypothesis about the bug:")


def test_construct_hypothesises_context_marks_refuted_and_current():
    fake = SimpleNamespace(hypothesises=["h1", "h2", "h3"])
    out = BaseAgent.construct_hypothesises_context(fake)
    assert "(Refuted) h1" in out
    assert "(Refuted) h2" in out
    assert "(Current hypothesis) h3" in out
    # only the last is current
    assert "(Refuted) h3" not in out


def test_construct_read_files_context_empty():
    out = BaseAgent.construct_read_files_context(SimpleNamespace(read_files={}))
    assert "No files have been read so far." in out


def test_construct_read_files_context_populated():
    fake = SimpleNamespace(read_files={"Foo.java": {"10,20": "CODE-BODY"}})
    out = BaseAgent.construct_read_files_context(fake)
    assert "Lines 10 to 20 from file: Foo.java" in out
    assert "CODE-BODY" in out


def test_construct_fixes_context_empty():
    out = BaseAgent.construct_fixes_context(SimpleNamespace(suggested_fixes=[]))
    assert "No fixes were suggested yet." in out


def test_construct_fixes_context_populated():
    fake = SimpleNamespace(suggested_fixes=[{"line": 1}, {"line": 2}])
    out = BaseAgent.construct_fixes_context(fake)
    assert "###Fix:" in out
    assert "{'line': 1}" in out
    assert "{'line': 2}" in out


def test_construct_search_context_empty():
    out = BaseAgent.construct_search_context(SimpleNamespace(search_queries=[]))
    assert "No search queries executed so far." in out


def test_construct_search_context_populated():
    fake = SimpleNamespace(search_queries=[{"query": ["kw1", "kw2"], "result": "RES"}])
    out = BaseAgent.construct_search_context(fake)
    assert "Searching keywords: ['kw1', 'kw2']" in out
    assert "RES" in out


def test_construct_extracted_methods_context_empty():
    out = BaseAgent.construct_extracted_methods_context(SimpleNamespace(extracted_methods=[]))
    assert "No extracted methods so far." in out


def test_construct_extracted_methods_context_populated():
    fake = SimpleNamespace(extracted_methods=[{"result": "METHOD-CODE-A"}, {"result": "METHOD-CODE-B"}])
    out = BaseAgent.construct_extracted_methods_context(fake)
    assert "METHOD-CODE-A" in out
    assert "METHOD-CODE-B" in out


def test_construct_similar_calls_context_empty():
    out = BaseAgent.construct_similar_calls_context(SimpleNamespace(similar_calls=None))
    assert "No similar functions calls were extracted." in out


def test_construct_similar_calls_context_populated():
    fake = SimpleNamespace(
        similar_calls=[{"code_snippet": "SNIP", "file_path": "F.java", "result": "CALLS"}]
    )
    out = BaseAgent.construct_similar_calls_context(fake)
    assert "Code snippet: SNIP" in out
    assert "target file: F.java" in out
    assert "CALLS" in out


def test_construct_bug_report_context_empty():
    out = BaseAgent.construct_bug_report_context(SimpleNamespace(bug_report={}))
    assert "No info was collected about the bug so far." in out


def test_construct_bug_report_context_populated():
    fake = SimpleNamespace(
        bug_report={"failing_test_code": "TEST-CODE"},
        localization_info="LOC#FAULT_OF_OMISSION-INFO",
        tests_results="TESTS-FAILED",
    )
    out = BaseAgent.construct_bug_report_context(fake)
    # the #FAULT_OF_OMISSION marker is stripped
    assert "#FAULT_OF_OMISSION" not in out
    assert "LOC-INFO" in out
    assert "TESTS-FAILED" in out
    assert "TEST-CODE" in out
    assert "The code of the failing test cases:" in out


def test_construct_bug_report_context_populated_no_failing_test_code():
    fake = SimpleNamespace(
        bug_report={"failing_test_code": ""},
        localization_info="LOC",
        tests_results="T",
    )
    out = BaseAgent.construct_bug_report_context(fake)
    assert "The code of the failing test cases:" not in out


def test_construct_commands_history_context_empty():
    out = BaseAgent.construct_commands_history_context(SimpleNamespace(commands_history=[]))
    # nothing appended when empty
    assert out == "## The list of commands you have executed so far:\n"


def test_construct_commands_history_context_populated():
    fake = SimpleNamespace(commands_history=["cmd-a", "cmd-b"])
    out = BaseAgent.construct_commands_history_context(fake)
    assert "cmd-a" in out and "cmd-b" in out


def test_construct_human_feedback_context_empty():
    out = BaseAgent.construct_human_feedback_context(SimpleNamespace(human_feedback=[]))
    assert out.startswith("## The list of human feedbacks:")


def test_construct_human_feedback_context_populated():
    fake = SimpleNamespace(human_feedback=["Human feedback: fix it"])
    out = BaseAgent.construct_human_feedback_context(fake)
    assert "Human feedback: fix it" in out


def test_construct_generated_methods_context_empty():
    out = BaseAgent.construct_generated_methods_context(SimpleNamespace(generated_methods=None))
    assert "No AI generated code yet." in out


def test_construct_generated_methods_context_populated():
    fake = SimpleNamespace(generated_methods=("myMethod", "GEN-BODY"))
    out = BaseAgent.construct_generated_methods_context(fake)
    assert "regeneration of method myMethod" in out
    assert "GEN-BODY" in out


# ===========================================================================
# construct_context_prompt : aggregates the context builders
# ===========================================================================
def test_construct_context_prompt_aggregates_sections():
    fake = SimpleNamespace(
        hypothesises=["h1"],
        read_files={},
        suggested_fixes=[],
        search_queries=[],
        bug_report={},
        commands_history=[],
        similar_calls=None,
        extracted_methods=[],
        generated_methods=None,
        history=[],
    )
    # bind the sub-methods that construct_context_prompt calls on self
    fake.construct_hypothesises_context = lambda: BaseAgent.construct_hypothesises_context(fake)
    fake.construct_read_files_context = lambda: BaseAgent.construct_read_files_context(fake)
    fake.construct_fixes_context = lambda: BaseAgent.construct_fixes_context(fake)
    fake.construct_search_context = lambda: BaseAgent.construct_search_context(fake)
    fake.construct_bug_report_context = lambda: BaseAgent.construct_bug_report_context(fake)
    fake.construct_commands_history_context = lambda: BaseAgent.construct_commands_history_context(fake)
    fake.construct_similar_calls_context = lambda: BaseAgent.construct_similar_calls_context(fake)
    fake.construct_extracted_methods_context = lambda: BaseAgent.construct_extracted_methods_context(fake)
    fake.construct_generated_methods_context = lambda: BaseAgent.construct_generated_methods_context(fake)
    fake.construct_unknown_commands = lambda: BaseAgent.construct_unknown_commands(fake)

    out = BaseAgent.construct_context_prompt(fake)
    assert "(Current hypothesis) h1" in out
    assert "No files have been read so far." in out
    assert "No fixes were suggested yet." in out
    assert "DO NOT TRY TO USE THE FOLLOWING COMMANDS" in out
    assert "This is the end of information sections." in out


# ===========================================================================
# construct_fix_query : aggregates several context builders + prompt_dictionary
# ===========================================================================
def test_construct_fix_query_includes_task_and_fix_format():
    fake = SimpleNamespace(
        hypothesises=[],
        read_files={},
        suggested_fixes=[],
        search_queries=[],
        bug_report={},
        similar_calls=None,
        prompt_dictionary={"fix format": ["FIX-FORMAT-LINE-1", "FIX-FORMAT-LINE-2"]},
    )
    fake.construct_hypothesises_context = lambda: BaseAgent.construct_hypothesises_context(fake)
    fake.construct_read_files_context = lambda: BaseAgent.construct_read_files_context(fake)
    fake.construct_fixes_context = lambda: BaseAgent.construct_fixes_context(fake)
    fake.construct_search_context = lambda: BaseAgent.construct_search_context(fake)
    fake.construct_bug_report_context = lambda: BaseAgent.construct_bug_report_context(fake)
    fake.construct_similar_calls_context = lambda: BaseAgent.construct_similar_calls_context(fake)

    out = BaseAgent.construct_fix_query(fake)
    assert "Suggest a list of 10 possible fixes" in out
    assert "FIX-FORMAT-LINE-1" in out
    assert "FIX-FORMAT-LINE-2" in out
    assert "cannot gather any more info" in out


# ===========================================================================
# History-driven construct_* builders
# ===========================================================================
def test_construct_generated_methods_pairs_with_next_result():
    cmd = assistant_cmd(
        "AI_generate_method_code",
        {"project_name": "Lang", "bug_index": 1, "filepath": "Foo.java", "method_name": "doIt"},
    )
    history = [Msg("assistant", cmd), Msg("user", "GENERATED-CODE-RESULT")]
    fake = bind_validate(SimpleNamespace(history=history, generated_methods=None))
    BaseAgent.construct_generated_methods(fake)
    assert fake.generated_methods == ("doIt", "GENERATED-CODE-RESULT")


def test_construct_generated_methods_none_when_no_command():
    history = [Msg("assistant", assistant_cmd("express_hypothesis", {"hypothesis": "h"}))]
    fake = bind_validate(SimpleNamespace(history=history, generated_methods=None))
    BaseAgent.construct_generated_methods(fake)
    assert fake.generated_methods is None


def test_construct_commands_history_records_name_and_reasoning():
    history = [
        Msg("assistant", assistant_cmd("run_test", {}, thoughts="because reasons")),
        Msg("assistant", assistant_cmd("read_range", {}, thoughts="reading now")),
    ]
    fake = SimpleNamespace(history=history)
    BaseAgent.construct_commands_history(fake)
    assert len(fake.commands_history) == 2
    assert "run_test" in fake.commands_history[0]
    assert "because reasons" in fake.commands_history[0]
    assert "read_range" in fake.commands_history[1]


def test_construct_commands_history_skips_repairagent_marker():
    marker = "## As RepairAgentv0.5.0, this is the last command I have called in response to the users' input"
    history = [Msg("assistant", marker + " ...")]
    fake = SimpleNamespace(history=history)
    BaseAgent.construct_commands_history(fake)
    assert fake.commands_history == []


def test_construct_suggested_fixes_collects_write_and_tryfixes():
    write_cmd = assistant_cmd(
        "write_fix",
        {"project_name": "Lang", "bug_index": 1, "changes_dicts": {"line": 1}},
    )
    try_cmd = assistant_cmd(
        "try_fixes",
        {"fixes_list": [{"changes_dicts": {"line": 2}}, {"changes_dicts": {"line": 3}}]},
    )
    history = [Msg("assistant", write_cmd), Msg("assistant", try_cmd)]
    fake = bind_validate(SimpleNamespace(history=history, suggested_fixes=None))
    BaseAgent.construct_suggested_fixes(fake)
    assert {"line": 1} in fake.suggested_fixes
    assert {"line": 2} in fake.suggested_fixes
    assert {"line": 3} in fake.suggested_fixes


def test_construct_suggested_fixes_empty_without_fix_commands():
    history = [Msg("assistant", assistant_cmd("run_test", {}))]
    fake = bind_validate(SimpleNamespace(history=history, suggested_fixes=None))
    BaseAgent.construct_suggested_fixes(fake)
    assert fake.suggested_fixes == []


def test_construct_human_feedback_collects_system_feedback():
    history = [
        Msg("system", "Human feedback: do better"),
        Msg("system", "some other system msg"),
        Msg("assistant", "x"),
    ]
    fake = SimpleNamespace(history=history)
    BaseAgent.construct_human_feedback(fake)
    assert fake.human_feedback == ["Human feedback: do better"]


def test_construct_unknown_commands_collects_flagged_commands():
    history = [
        Msg("assistant", assistant_cmd("weird_cmd", {})),
        Msg("user", "weird_cmd is an unknown command. Do not try to use this command again."),
    ]
    fake = SimpleNamespace(history=history)
    out = BaseAgent.construct_unknown_commands(fake)
    assert out == ["weird_cmd"]


def test_construct_unknown_commands_ignores_when_not_flagged():
    history = [
        Msg("assistant", assistant_cmd("run_test", {})),
        Msg("user", "fine, command executed"),
    ]
    fake = SimpleNamespace(history=history)
    assert BaseAgent.construct_unknown_commands(fake) == []


def test_construct_search_queries_pairs_keywords_and_result():
    cmd = assistant_cmd(
        "search_code_base",
        {"project_name": "Lang", "bug_index": 1, "key_words": ["foo", "bar"]},
    )
    history = [Msg("assistant", cmd), Msg("user", "SEARCH-RESULT")]
    fake = bind_validate(SimpleNamespace(history=history, search_queries=None))
    BaseAgent.construct_search_queries(fake)
    assert fake.search_queries == [{"query": ["foo", "bar"], "result": "SEARCH-RESULT"}]


def test_construct_similar_calls_pairs_snippet_and_result():
    cmd = assistant_cmd(
        "extract_similar_functions_calls",
        {"project_name": "Lang", "bug_index": 1, "file_path": "F.java", "code_snippet": "SNIP"},
    )
    history = [Msg("assistant", cmd), Msg("user", "SIMILAR-RESULT")]
    fake = bind_validate(SimpleNamespace(history=history, similar_calls=None))
    BaseAgent.construct_similar_calls(fake)
    assert fake.similar_calls == [
        {"code_snippet": "SNIP", "file_path": "F.java", "result": "SIMILAR-RESULT"}
    ]


def test_construct_extracted_methods_pairs_method_and_result():
    cmd = assistant_cmd(
        "extract_method_code",
        {"project_name": "Lang", "bug_index": 1, "filepath": "F.java", "method_name": "m"},
    )
    history = [Msg("assistant", cmd), Msg("user", "EXTRACT-RESULT")]
    fake = bind_validate(SimpleNamespace(history=history, extracted_methods=None))
    BaseAgent.construct_extracted_methods(fake)
    assert fake.extracted_methods == [
        {"method_name": "m", "file_path": "F.java", "result": "EXTRACT-RESULT"}
    ]


def test_construct_bug_report_collects_failing_test_code():
    cmd = assistant_cmd(
        "extract_test_code",
        {"project_name": "Lang", "bug_index": 1, "test_file_path": "T.java"},
    )
    history = [Msg("assistant", cmd), Msg("user", "FAILING-TEST-BODY")]
    fake = bind_validate(SimpleNamespace(history=history, bug_report=None))
    BaseAgent.construct_bug_report(fake)
    assert "FAILING-TEST-BODY" in fake.bug_report["failing_test_code"]
    assert "T.java" in fake.bug_report["failing_test_code"]


def test_construct_hypothesises_collects_expressed_hypotheses():
    cmd1 = assistant_cmd("express_hypothesis", {"hypothesis": "first"})
    cmd2 = assistant_cmd("express_hypothesis", {"hypothesis": "second"})
    history = [Msg("assistant", cmd1), Msg("assistant", cmd2)]
    fake = bind_validate(SimpleNamespace(history=history, hypothesises=None))
    BaseAgent.construct_hypothesises(fake)
    assert fake.hypothesises == ["first", "second"]


# ===========================================================================
# response_format_instruction
# ===========================================================================
def test_response_format_instruction_with_command_format():
    fake = SimpleNamespace(
        config=SimpleNamespace(openai_functions=False),
        command_registry=SimpleNamespace(commands={}),
    )
    out = BaseAgent.response_format_instruction(fake, "one-shot")
    assert "Respond strictly with JSON" in out
    assert "interface Response" in out
    # WITH_COMMAND format includes a command field example
    assert '"command"' in out


def test_response_format_instruction_without_command_when_functions_enabled():
    fake = SimpleNamespace(
        config=SimpleNamespace(openai_functions=True),
        command_registry=SimpleNamespace(commands={"x": object()}),
    )
    out = BaseAgent.response_format_instruction(fake, "one-shot")
    # functions enabled + commands present -> mentions function_call
    assert "function_call" in out
    assert "criticism" in out  # WITHOUT_COMMAND template


def test_response_format_instruction_rejects_unknown_process():
    fake = SimpleNamespace(config=SimpleNamespace(openai_functions=False))
    with pytest.raises(NotImplementedError):
        BaseAgent.response_format_instruction(fake, "multi-shot")


# ===========================================================================
# save_to_json
# ===========================================================================
def test_save_to_json_new_file_with_list(tmp_path):
    path = str(tmp_path / "out.json")
    content = [{"a": 1}, {"b": 2}]
    ret = BaseAgent.save_to_json(SimpleNamespace(), path, content)
    assert ret == content
    assert json.load(open(path)) == content


def test_save_to_json_new_file_with_plain_dict_wraps_in_list(tmp_path):
    path = str(tmp_path / "out.json")
    content = {"name": "fix-1"}
    ret = BaseAgent.save_to_json(SimpleNamespace(), path, content)
    assert ret == [content]
    assert json.load(open(path)) == [content]


def test_save_to_json_new_file_with_fixes_dict_flattens(tmp_path):
    path = str(tmp_path / "out.json")
    content = {"fixes_list": [{"f": 1}, {"f": 2}]}
    ret = BaseAgent.save_to_json(SimpleNamespace(), path, content)
    assert ret == [{"f": 1}, {"f": 2}]
    assert json.load(open(path)) == [{"f": 1}, {"f": 2}]


def test_save_to_json_appends_list_to_existing_file(tmp_path):
    path = str(tmp_path / "out.json")
    json.dump([{"a": 1}], open(path, "w"))
    ret = BaseAgent.save_to_json(SimpleNamespace(), path, [{"b": 2}])
    assert ret == [{"b": 2}]
    assert json.load(open(path)) == [{"a": 1}, {"b": 2}]


def test_save_to_json_appends_dict_to_existing_file(tmp_path):
    path = str(tmp_path / "out.json")
    json.dump([{"a": 1}], open(path, "w"))
    ret = BaseAgent.save_to_json(SimpleNamespace(), path, {"b": 2})
    assert ret == [{"b": 2}]
    assert json.load(open(path)) == [{"a": 1}, {"b": 2}]


def test_save_to_json_appends_fixes_dict_flattened_to_existing(tmp_path):
    path = str(tmp_path / "out.json")
    json.dump([{"a": 1}], open(path, "w"))
    ret = BaseAgent.save_to_json(SimpleNamespace(), path, {"mutations": [{"m": 1}, {"m": 2}]})
    assert ret == [{"m": 1}, {"m": 2}]
    assert json.load(open(path)) == [{"a": 1}, {"m": 1}, {"m": 2}]


# ===========================================================================
# construct_mutation_prompt
# ===========================================================================
def _mutation_self(populated):
    """Build a fake self for construct_mutation_prompt.

    When ``populated`` is True the context sections carry real data so they are
    appended to info_sections; when False every section reports 'No ...' and is
    skipped.
    """
    if populated:
        fake = SimpleNamespace(
            hypothesises=["h1"],
            read_files={"F.java": {"1,2": "BODY"}},
            suggested_fixes=[{"line": 1}],
            search_queries=[{"query": ["k"], "result": "R"}],
            bug_report={"failing_test_code": "TC"},
            commands_history=["c1"],
            similar_calls=[{"code_snippet": "S", "file_path": "F", "result": "RES"}],
            extracted_methods=[{"result": "M"}],
            localization_info="LOC",
            tests_results="T",
            project_name="Lang",
            bug_index="1",
            prompt_dictionary={"fix format": ["FMT"]},
        )
    else:
        fake = SimpleNamespace(
            hypothesises=[],
            read_files={},
            suggested_fixes=[],
            search_queries=[],
            bug_report={},
            commands_history=[],
            similar_calls=None,
            extracted_methods=[],
            localization_info="LOC",
            tests_results="T",
            project_name="Lang",
            bug_index="1",
            prompt_dictionary={"fix format": ["FMT"]},
        )
    fake.construct_hypothesises_context = lambda: BaseAgent.construct_hypothesises_context(fake)
    fake.construct_read_files_context = lambda: BaseAgent.construct_read_files_context(fake)
    fake.construct_fixes_context = lambda: BaseAgent.construct_fixes_context(fake)
    fake.construct_search_context = lambda: BaseAgent.construct_search_context(fake)
    fake.construct_bug_report_context = lambda: BaseAgent.construct_bug_report_context(fake)
    fake.construct_commands_history_context = lambda: BaseAgent.construct_commands_history_context(fake)
    fake.construct_similar_calls_context = lambda: BaseAgent.construct_similar_calls_context(fake)
    fake.construct_extracted_methods_context = lambda: BaseAgent.construct_extracted_methods_context(fake)
    return fake


def test_construct_mutation_prompt_includes_template_hints_and_task(monkeypatch):
    monkeypatch.setattr(base_mod, "create_fix_template", lambda p, b: "FIX-TEMPLATE")
    fake = _mutation_self(populated=True)
    out = BaseAgent.construct_mutation_prompt(fake, "LAST-PATCH", "DETAILED-BUGGIES")
    assert "FIX-TEMPLATE" in out
    assert "DETAILED-BUGGIES" in out
    assert "generate 30 mutants" in out
    assert "FMT" in out
    # populated sections are included
    assert "BODY" in out
    assert "TC" in out


def test_construct_mutation_prompt_skips_empty_sections(monkeypatch):
    monkeypatch.setattr(base_mod, "create_fix_template", lambda p, b: "FIX-TEMPLATE")
    fake = _mutation_self(populated=False)
    out = BaseAgent.construct_mutation_prompt(fake, "LAST-PATCH", "DETAILED-BUGGIES")
    # the empty-marker sentences must NOT leak into the prompt
    assert "No files have been read so far." not in out
    assert "No fixes were suggested yet." not in out
    assert "FIX-TEMPLATE" in out
    assert "generate 30 mutants" in out


# ===========================================================================
# save_context
# ===========================================================================
def test_save_context_writes_serialized_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp_name = "exp_save_ctx"
    out_dir = tmp_path / "experimental_setups" / exp_name / "saved_contexts"
    out_dir.mkdir(parents=True)

    history = [Msg("user", "hi"), Msg("assistant", "yo")]
    fake = SimpleNamespace(
        cycle_budget=4,
        cycle_count=1,
        cycles_remaining=3,
        current_state="collect information to understand the bug",
        prompt_dictionary={"role": "r"},
        project_name="Lang",
        bug_index="2",
        localization_info="LOC",
        tests_results="TESTS",
        read_files={"F": {"1,2": "x"}},
        suggested_fixes=[{"a": 1}],
        search_queries=[],
        bug_report={"failing_test_code": "tc"},
        commands_history=["c1"],
        human_feedback=[],
        ask_chatgpt=None,
        hypothesises=["h"],
        initial_bug_report={},
        buggy_lines="lines",
        similar_calls=None,
        extracted_methods=[],
        experiment_file="exp.json",
        hyperparams={"x": 1},
        history=history,
        exps=[exp_name],
    )
    BaseAgent.save_context(fake)

    ctx_file = out_dir / "saved_context_Lang_2"
    saved = json.load(open(ctx_file))
    assert saved["cycle_budget"] == 4
    assert saved["project_name"] == "Lang"
    assert saved["test_results"] == "TESTS"
    # history is serialized as role/content dicts
    assert saved["history"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


# ===========================================================================
# load_context
# ===========================================================================
def test_load_context_restores_all_attributes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp_name = "exp_load_ctx"
    ctx_dir = tmp_path / "experimental_setups" / exp_name / "saved_contexts"
    ctx_dir.mkdir(parents=True)
    project_name, bug_index = "Lang", "3"
    saved = {
        "cycle_budget": 5,
        "cycle_count": 2,
        "cycle_remaining": 3,
        "current_state": "collect information to fix the bug",
        "prompt_dictionary": {"role": "r"},
        "project_name": project_name,
        "bug_index": bug_index,
        "localization_info": "LOC",
        "test_results": "TESTS",
        "read_files": {"F": {"1,2": "x"}},
        "suggested_fixes": [{"a": 1}],
        "search_queries": [{"query": ["k"], "result": "r"}],
        "bug_report": {"failing_test_code": "tc"},
        "commands_history": ["c1"],
        "human_feedback": ["hf"],
        "ask_chatgpt": None,
        "hypothesises": ["h"],
        "initial_bug_report": {},
        "buggy_lines": "lines",
        "similar_calls": [],
        "extracted_methods": [],
        "experiment_file": "exp.json",
        "hyperparams": {"x": 1},
        "history": [{"role": "user", "content": "hi"}],
    }
    ctx_file = ctx_dir / f"saved_context_{project_name}_{bug_index}"
    json.dump(saved, open(ctx_file, "w"))

    fake = SimpleNamespace(
        exps=[exp_name], project_name=project_name, bug_index=bug_index
    )
    BaseAgent.load_context(fake)
    assert fake.cycle_budget == 5
    assert fake.cycle_count == 2
    assert fake.cycles_remaining == 3
    assert fake.current_state == "collect information to fix the bug"
    assert fake.read_files == {"F": {"1,2": "x"}}
    assert fake.test_results if hasattr(fake, "test_results") else fake.tests_results == "TESTS"
    assert fake.hyperparams == {"x": 1}
    assert fake.history == [{"role": "user", "content": "hi"}]


# ===========================================================================
# on_before_think
# ===========================================================================
def test_on_before_think_returns_prompt_unchanged_without_plugins():
    model_name = "gpt-4"
    prompt = ChatSequence.for_model(model_name, [Message("system", "sys"), Message("user", "u")])
    fake = SimpleNamespace(
        config=SimpleNamespace(plugins=[]),
        ai_config=SimpleNamespace(prompt_generator=object()),
        llm=SimpleNamespace(name=model_name),
        send_token_limit=100000,
    )
    out = BaseAgent.on_before_think(fake, prompt, "one-shot", "instr")
    assert out is prompt
    assert len(out) == 2


def test_on_before_think_inserts_capable_plugin_response():
    model_name = "gpt-4"
    prompt = ChatSequence.for_model(model_name, [Message("system", "sys"), Message("user", "u")])

    class Plugin:
        def can_handle_on_planning(self):
            return True

        def on_planning(self, prompt_generator, raw):
            return "PLUGIN-PLANNING-OUTPUT"

    fake = SimpleNamespace(
        config=SimpleNamespace(plugins=[Plugin()]),
        ai_config=SimpleNamespace(prompt_generator=object()),
        llm=SimpleNamespace(name=model_name),
        send_token_limit=100000,
    )
    out = BaseAgent.on_before_think(fake, prompt, "one-shot", "instr")
    contents = [m.content for m in out]
    assert "PLUGIN-PLANNING-OUTPUT" in contents
    # inserted before the last (cycle instruction) message
    assert contents[-1] == "u"


def test_on_before_think_skips_incapable_plugin():
    model_name = "gpt-4"
    prompt = ChatSequence.for_model(model_name, [Message("system", "sys"), Message("user", "u")])

    class Plugin:
        def can_handle_on_planning(self):
            return False

        def on_planning(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("should not be called")

    fake = SimpleNamespace(
        config=SimpleNamespace(plugins=[Plugin()]),
        ai_config=SimpleNamespace(prompt_generator=object()),
        llm=SimpleNamespace(name=model_name),
        send_token_limit=100000,
    )
    out = BaseAgent.on_before_think(fake, prompt, "one-shot", "instr")
    assert len(out) == 2


def test_on_before_think_stops_when_token_limit_exceeded():
    model_name = "gpt-4"
    prompt = ChatSequence.for_model(model_name, [Message("system", "sys"), Message("user", "u")])

    class Plugin:
        def can_handle_on_planning(self):
            return True

        def on_planning(self, prompt_generator, raw):
            return "X" * 50

    fake = SimpleNamespace(
        config=SimpleNamespace(plugins=[Plugin()]),
        ai_config=SimpleNamespace(prompt_generator=object()),
        llm=SimpleNamespace(name=model_name),
        send_token_limit=0,  # any addition exceeds the limit -> break
    )
    out = BaseAgent.on_before_think(fake, prompt, "one-shot", "instr")
    assert len(out) == 2  # nothing inserted


# ===========================================================================
# add_history_upto_token_limit (module-level function)
# ===========================================================================
def _history_with_one_cycle(model_name):
    history = MessageHistory(SimpleNamespace(name=model_name))
    # a full cycle: user input, ai_response (valid JSON), action_result
    history.append(Message("user", "do the thing"))
    history.append(Message("assistant", assistant_cmd("run_test", {}), "ai_response"))
    history.append(Message("user", "the result", "action_result"))
    return history


def test_add_history_upto_token_limit_inserts_when_under_limit():
    model_name = "gpt-4"
    prompt = ChatSequence.for_model(model_name, [Message("system", "sys")])
    history = _history_with_one_cycle(model_name)
    trimmed = add_history_upto_token_limit(prompt, history, t_limit=100000)
    assert trimmed == []
    # the cycle's three messages got inserted into the prompt
    assert len(prompt) == 4


def test_add_history_upto_token_limit_trims_when_over_limit():
    model_name = "gpt-4"
    prompt = ChatSequence.for_model(model_name, [Message("system", "sys")])
    history = _history_with_one_cycle(model_name)
    trimmed = add_history_upto_token_limit(prompt, history, t_limit=1)
    # nothing inserted; the cycle's messages are returned as trimmed
    assert len(prompt) == 1
    assert len(trimmed) == 3
