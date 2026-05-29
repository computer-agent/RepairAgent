"""Additional unit tests for ``autogpt.commands.defects4j`` aimed at raising
coverage of the many functions NOT already covered by ``test_defects4j_logic``
(parse_buggy_lines, create_fix_template, get_edited_files, extract_file_name,
extract_root_cause, list_java_files) or ``test_preprocess_paths`` (preprocess_paths).

Strategy:
* Pure / parsing helpers are exercised directly.
* Subprocess / OS / LLM backed functions are covered by monkeypatching the
  symbols *inside the defects4j module namespace* (never running Java, git,
  defects4j, or the network).
* File/agent backed functions build a fake ``<workspace>/<proj>_<idx>_buggy``
  tree (via the ``buggy_project_dir`` fixture or a tmp tree) and mock nested
  calls (run_defects4j_tests, extract_fail_report, LLM, etc.).
"""

import os
import json
from types import SimpleNamespace

import logging

import pytest

import tests.conftest  # noqa: F401  (sys.path / chdir / langchain stubs)

from autogpt.commands import defects4j as d
from autogpt.logs import logger as _project_logger


@pytest.fixture
def quiet_info_logging():
    """Raise the project logger above INFO for the duration of a test.

    ``extract_function_calls`` does ``logger.info(list(...))``. The project's
    log formatter (``remove_color_codes``) assumes the message is a ``str`` and
    raises ``TypeError`` when handed a ``list`` (a real latent bug, see report).
    Whether it surfaces depends on the active log level, so tests that drive
    these paths suppress INFO records first to assert the function's behavior
    independently of the logging defect.
    """
    underlying = _project_logger.logger
    previous = underlying.level
    underlying.setLevel(logging.WARNING)
    try:
        yield
    finally:
        underlying.setLevel(previous)


# ===========================================================================
# create_deletion_template
# ===========================================================================

def _make_buggy_lines(base_dir, name, index, content):
    bdir = os.path.join(base_dir, "defects4j", "buggy-lines")
    os.makedirs(bdir, exist_ok=True)
    path = os.path.join(bdir, "{}-{}.buggy.lines".format(name, index))
    with open(path, "w") as fh:
        fh.write(content)
    return path


def test_create_deletion_template_builds_deletions(tmp_path, monkeypatch):
    _make_buggy_lines(
        str(tmp_path), "Lang", "1",
        "src/Foo.java#10#int x;\nsrc/Foo.java#12#int y;\nsrc/Bar.java#5#z;\n",
    )
    monkeypatch.chdir(tmp_path)
    out = d.create_deletion_template("Lang", "1")
    assert isinstance(out, list)
    by_file = {entry["file_name"]: entry for entry in out}
    assert by_file["src/Foo.java"]["deletions"] == ["10", "12"]
    assert by_file["src/Bar.java"]["deletions"] == ["5"]
    # other buckets are empty placeholders
    assert by_file["src/Foo.java"]["insertions"] == []
    assert by_file["src/Foo.java"]["modifications"] == []


def test_create_deletion_template_returns_none_on_fault_of_omission(tmp_path, monkeypatch):
    _make_buggy_lines(
        str(tmp_path), "Lang", "2",
        "src/Foo.java#10#FAULT_OF_OMISSION\n",
    )
    monkeypatch.chdir(tmp_path)
    assert d.create_deletion_template("Lang", "2") is None


# ===========================================================================
# we_are_running_in_a_docker_container : always True (early return)
# ===========================================================================

def test_we_are_running_in_a_docker_container_always_true():
    assert d.we_are_running_in_a_docker_container() is True


# ===========================================================================
# extract_function_calls
# ===========================================================================

def test_extract_function_calls_basic():
    code = "void m() { foo(1); bar(a, b); }"
    calls = d.extract_function_calls(code)
    joined = " ".join(calls)
    assert "foo(1)" in joined
    assert "bar(a, b)" in joined


def test_extract_function_calls_strips_comments_first():
    # the call inside a // comment is removed before matching
    code = "x(); // hidden(99)\n"
    calls = d.extract_function_calls(code)
    assert any("x()" in c for c in calls)
    assert not any("hidden" in c for c in calls)


def test_extract_function_calls_none(quiet_info_logging):
    assert d.extract_function_calls("int x = 1;") == []


# ===========================================================================
# extract_targeted_lines & get_list_of_buggy_lines edge / missing-file
# (logic file covers the happy path; cover the missing-file branch here)
# ===========================================================================

def test_get_list_of_buggy_lines_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert d.get_list_of_buggy_lines("Nope", "99") == []


def test_get_list_of_buggy_lines_parses_line_numbers(tmp_path, monkeypatch):
    _make_buggy_lines(
        str(tmp_path), "Lang", "3",
        "src/Foo.java#10#int x;\nsrc/Foo.java#22#int y;\n",
    )
    monkeypatch.chdir(tmp_path)
    assert d.get_list_of_buggy_lines("Lang", "3") == [10, 22]


# ===========================================================================
# get_localization
# ===========================================================================

def test_get_localization_no_files_returns_blank(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = d.get_localization("Lang", "99")
    assert out == "\n"


def test_get_localization_with_lines_and_methods(tmp_path, monkeypatch):
    _make_buggy_lines(
        str(tmp_path), "Lang", "1",
        "src/Foo.java#10#int x;\n",
    )
    mdir = os.path.join(str(tmp_path), "defects4j", "buggy-methods")
    os.makedirs(mdir, exist_ok=True)
    # only lines ending with "1" are included
    with open(os.path.join(mdir, "Lang-1.buggy.methods"), "w") as fh:
        fh.write("org.foo.Bar#method1\norg.foo.Bar#method2\n")
    monkeypatch.chdir(tmp_path)
    out = d.get_localization("Lang", "1")
    assert "located at exactly these lines" in out
    assert "src/Foo.java#10#int x;" in out
    assert "list of buggy methods" in out
    assert "method1" in out
    assert "method2" not in out


# ===========================================================================
# apply_changes : deletions / modifications / insertions on a real tmp file
# ===========================================================================

def test_apply_changes_deletion_blanks_line(tmp_path):
    f = tmp_path / "F.java"
    f.write_text("a\nb\nc\n")
    d.apply_changes({"file_name": str(f), "deletions": ["2"]})
    assert f.read_text() == "a\n\nc\n"


def test_apply_changes_modification_similar_line_replaced(tmp_path):
    f = tmp_path / "F.java"
    f.write_text("int x = 1;\n")
    d.apply_changes({
        "file_name": str(f),
        "modifications": [{"line_number": 1, "modified_line": "int x = 2;"}],
    })
    assert f.read_text() == "int x = 2;\n"


def test_apply_changes_modification_dissimilar_skipped(tmp_path):
    # fuzz ratio < 70 -> the modification is skipped, original kept
    f = tmp_path / "F.java"
    f.write_text("int x = 1;\n")
    d.apply_changes({
        "file_name": str(f),
        "modifications": [{"line_number": 1,
                           "modified_line": "totally different content here zzzzz"}],
    })
    assert f.read_text() == "int x = 1;\n"


def test_apply_changes_insertion_adds_lines(tmp_path):
    f = tmp_path / "F.java"
    f.write_text("a\nb\n")
    d.apply_changes({
        "file_name": str(f),
        "insertions": [{"line_number": 2, "new_lines": ["x\n", "y\n"]}],
    })
    # inserted before original line 2
    assert f.read_text() == "a\nx\ny\nb\n"


# ===========================================================================
# extract_fail_report : parse the failing_tests file
# ===========================================================================

def test_extract_fail_report_groups_cases(buggy_project_dir):
    b = buggy_project_dir
    content = (
        "--- org.foo.BarTest::testA\n"
        "junit.framework.AssertionFailedError\n"
        "\tat org.foo.BarTest.testA(BarTest.java:10)\n"
        "\tat sun.unrelated.Frame(Other.java:1)\n"
        "--- org.foo.BazTest::testB\n"
        "some message\n"
    )
    (b.project_dir / "failing_tests").write_text(content)
    out = d.extract_fail_report(b.project_name, b.bug_index, b.agent)
    assert "There are 2 failing test cases" in out
    # only stack frames matching the case base are kept
    assert "org.foo.BarTest.testA(BarTest.java:10)" in out
    assert "sun.unrelated.Frame" not in out
    assert "org.foo.BazTest::testB" in out


def test_extract_fail_report_missing_separator_raises(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "failing_tests").write_text("--- no double colon here\n")
    with pytest.raises(ValueError):
        d.extract_fail_report(b.project_name, b.bug_index, b.agent)


# ===========================================================================
# list_files : recursive .java collection
# ===========================================================================

def test_list_files_only_java(tmp_path):
    (tmp_path / "A.java").write_text("class A{}")
    (tmp_path / "b.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "C.java").write_text("class C{}")
    result = d.list_files(str(tmp_path))
    bases = sorted(os.path.basename(p) for p in result)
    assert bases == ["A.java", "C.java"]


def test_list_files_empty(tmp_path):
    assert d.list_files(str(tmp_path)) == []


# ===========================================================================
# get_classes_and_methods : javalang parsing on a real file
# ===========================================================================

def _write_java_project(b, rel, code):
    """Write java file under project dir and create a files_index.txt so that
    preprocess_paths resolves the dotted path to ``rel`` directly."""
    (b.project_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (b.project_dir / rel).write_text(code)
    return rel


def test_get_classes_and_methods_returns_class_methods(buggy_project_dir):
    b = buggy_project_dir
    code = (
        "public class Foo {\n"
        "    public int bar(int x) { return x; }\n"
        "    public void baz() {}\n"
        "}\n"
    )
    # put under source/ so the source_dir branch is hit, and store file so that
    # preprocess_paths finds it directly (path exists).
    (b.project_dir / "source").mkdir()
    _write_java_project(b, "Foo.java", code)
    out = d.get_classes_and_methods(b.project_name, b.bug_index, "Foo.java", b.agent)
    assert "Foo" in out
    assert "bar" in out
    assert "baz" in out


# ===========================================================================
# search_code_base : keyword match against method names
# ===========================================================================

def test_search_code_base_matches_method_names(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "src").mkdir()
    (b.project_dir / "src" / "Foo.java").write_text(
        "public class Foo {\n"
        "    public void computeValue() {}\n"
        "    public void other() {}\n"
        "}\n"
    )
    out = d.search_code_base(b.project_name, b.bug_index, ["compute"], b.agent)
    assert "matches were found" in out
    assert "computeValue" in out
    # "other" does not match the keyword
    assert "'other'" not in out


def test_search_code_base_no_match(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "src").mkdir()
    (b.project_dir / "src" / "Foo.java").write_text(
        "public class Foo { public void aaa() {} }\n"
    )
    out = d.search_code_base(b.project_name, b.bug_index, ["zzzzz"], b.agent)
    assert "matches were found" in out
    # empty match dict
    assert "{}" in out


# ===========================================================================
# execute_read_range / read_range
# ===========================================================================

def test_execute_read_range_returns_numbered_lines(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "F.java").write_text("l1\nl2\nl3\nl4\n")
    out = d.execute_read_range(b.project_name, b.bug_index, "F.java", 2, 3, b.agent)
    assert "Line 2:l2" in out
    assert "Line 3:l3" in out
    assert "Line 1:" not in out


def test_execute_read_range_eof(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "F.java").write_text("l1\nl2\n")
    out = d.execute_read_range(b.project_name, b.bug_index, "F.java", 1, 10, b.agent)
    assert out.endswith("EOF")


def test_read_range_delegates(buggy_project_dir, monkeypatch):
    sentinel = "READ_RESULT"
    monkeypatch.setattr(d, "execute_read_range", lambda *a, **k: sentinel)
    b = buggy_project_dir
    assert d.read_range(b.project_name, b.bug_index, "F.java", 1, 2, b.agent) == sentinel


# ===========================================================================
# execute_write_range / write_range : apply changes then run tests (mocked)
# ===========================================================================

def test_execute_write_range_applies_and_runs_tests(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    (b.project_dir / "F.java").write_text("int x = 1;\n")
    monkeypatch.setattr(d, "run_defects4j_tests", lambda *a, **k: "0 failing test cases")
    changes = [{
        "file_name": "F.java",
        "modifications": [{"line_number": 1, "modified_line": "int x = 2;"}],
        "deletions": [],
        "insertions": [],
    }]
    out = d.execute_write_range(b.project_name, b.bug_index, changes, b.agent)
    assert "Lines written successfully" in out
    assert "0 failing test cases" in out
    assert (b.project_dir / "F.java").read_text() == "int x = 2;\n"


def test_write_range_delegates(buggy_project_dir, monkeypatch):
    monkeypatch.setattr(d, "execute_write_range", lambda *a, **k: "WROTE")
    b = buggy_project_dir
    assert d.write_range(b.project_name, b.bug_index, [], b.agent) == "WROTE"


# ===========================================================================
# write_fix
# ===========================================================================

def test_write_fix_empty_changes(buggy_project_dir):
    b = buggy_project_dir
    b.agent.dummy_fix = True
    out = d.write_fix(b.project_name, b.bug_index, [], b.agent)
    assert "empty" in out.lower()


def test_write_fix_missed_buggy_lines_returns_template(buggy_project_dir, monkeypatch, tmp_path):
    b = buggy_project_dir
    b.agent.dummy_fix = True  # skip the deletion-dummy-fix branch
    # buggy line 10 exists but the fix targets line 99 -> missed lines triggers template
    _make_buggy_lines(str(b.workspace), "Lang", "1", "F.java#10#int x;\n")
    monkeypatch.chdir(b.workspace)
    changes = [{
        "file_name": "F.java",
        "modifications": [{"line_number": "99", "modified_line": "y"}],
        "deletions": [], "insertions": [],
    }]
    out = d.write_fix("Lang", 1, changes, b.agent)
    assert "did not target all the buggy lines" in out
    assert "[10]" in out


def test_write_fix_dummy_deletion_fixes_all(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    b.agent.dummy_fix = False
    # deletion template will be non-None
    monkeypatch.setattr(d, "create_deletion_template",
                        lambda *a, **k: [{"file_name": "F.java", "deletions": ["1"]}])
    monkeypatch.setattr(d, "get_list_of_buggy_lines", lambda *a, **k: [])
    monkeypatch.setattr(d, "execute_write_range", lambda *a, **k: "result: 0 failing test cases")
    out = d.write_fix(b.project_name, b.bug_index, [{"file_name": "F.java"}], b.agent)
    assert "Deleting the buggy lines fixed the problem" in out
    assert b.agent.dummy_fix is True


def test_write_fix_success_path(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    b.agent.dummy_fix = True
    monkeypatch.setattr(d, "get_list_of_buggy_lines", lambda *a, **k: [])
    monkeypatch.setattr(d, "extract_targeted_lines", lambda *a, **k: [])
    monkeypatch.setattr(d, "execute_write_range", lambda *a, **k: "test results")
    out = d.write_fix(b.project_name, b.bug_index, [{"file_name": "F.java"}], b.agent)
    assert "test results" in out
    assert "trying out candidate fixes" in out


# ===========================================================================
# try_fixes
# ===========================================================================

def test_try_fixes_empty_list():
    out = d.try_fixes("Lang", 1, [], agent=SimpleNamespace())
    assert "empty" in out.lower()


def test_try_fixes_list_of_dicts(monkeypatch):
    # when fixes_list[0] is a dict, the whole list is written at once
    monkeypatch.setattr(d, "execute_write_range", lambda *a, **k: "0 failing test cases")
    out = d.try_fixes("Lang", 1, [{"file_name": "F.java"}], agent=SimpleNamespace())
    assert "1 of them passed" in out
    assert "[0]" in out


def test_try_fixes_list_of_lists(monkeypatch):
    # when first element is not a dict, each fix is iterated
    calls = []

    def fake_write(pn, bi, fix, agent):
        calls.append(fix)
        return "0 failing test cases" if fix == ["good"] else "1 failing test cases"

    monkeypatch.setattr(d, "execute_write_range", fake_write)
    out = d.try_fixes("Lang", 1, [["bad"], ["good"]], agent=SimpleNamespace())
    assert "1 of them passed" in out
    assert "[1]" in out
    assert calls == [["bad"], ["good"]]


# ===========================================================================
# run_checkout / undo_changes / run_tests / run_defects4j_tests / execute_get_info
# (subprocess mocked at module namespace)
# ===========================================================================

class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_subprocess_run(monkeypatch, proc, recorder=None):
    def fake_run(*args, **kwargs):
        if recorder is not None:
            recorder.append((args, kwargs))
        return proc
    monkeypatch.setattr(d.subprocess, "run", fake_run)


def test_run_checkout_success(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=0))
    out = d.run_checkout(b.project_name, b.bug_index, b.agent)
    assert "restored to their original content" in out


def test_run_checkout_error(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=1, stderr="boom"))
    out = d.run_checkout(b.project_name, b.bug_index, b.agent)
    assert out == "Error: boom"


def test_run_checkout_removes_existing_workspace(make_fake_agent, monkeypatch, tmp_path):
    # run_checkout uses a repo-relative auto_gpt_workspace/<folder>
    monkeypatch.chdir(tmp_path)
    agent = make_fake_agent(str(tmp_path))
    folder = os.path.join("auto_gpt_workspace", "lang_1_buggy")
    os.makedirs(folder)
    (tmp_path / folder / "stale.txt").write_text("old")
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=0))
    d.run_checkout("Lang", 1, agent)
    # the stale workspace was removed before running checkout
    assert not os.path.exists(folder)


def test_undo_changes_delegates(buggy_project_dir, monkeypatch):
    monkeypatch.setattr(d, "run_checkout", lambda *a, **k: "UNDONE")
    b = buggy_project_dir
    assert d.undo_changes(b.project_name, b.bug_index, b.agent) == "UNDONE"


def test_run_tests_delegates(buggy_project_dir, monkeypatch):
    monkeypatch.setattr(d, "run_defects4j_tests", lambda *a, **k: "RAN")
    b = buggy_project_dir
    assert d.run_tests(b.project_name, b.bug_index, b.agent) == "RAN"


def test_run_defects4j_tests_build_failed(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=0, stdout="prefix BUILD FAILED tail"))
    monkeypatch.setattr(d, "undo_changes", lambda *a, **k: None)
    out = d.run_defects4j_tests(b.project_name, b.bug_index, b.agent)
    assert out.startswith("BUILD FAILED")
    # the per-run test file is created empty
    assert (b.workspace / "lang_1_buggy_test.txt").read_text() == ""


def test_run_defects4j_tests_success_calls_fail_report(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=0, stdout="OK normal output"))
    monkeypatch.setattr(d, "extract_fail_report", lambda *a, **k: "FAIL_REPORT")
    monkeypatch.setattr(d, "undo_changes", lambda *a, **k: None)
    out = d.run_defects4j_tests(b.project_name, b.bug_index, b.agent)
    assert out == "FAIL_REPORT"
    assert (b.workspace / "lang_1_buggy_test.txt").read_text() == "OK normal output"


def test_run_defects4j_tests_nonzero_build_failed(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=1, stderr="x BUILD FAILED y"))
    monkeypatch.setattr(d, "undo_changes", lambda *a, **k: None)
    out = d.run_defects4j_tests(b.project_name, b.bug_index, b.agent)
    assert out.startswith("BUILD FAILED")


def test_run_defects4j_tests_nonzero_other_error(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=2, stderr="some other error"))
    monkeypatch.setattr(d, "undo_changes", lambda *a, **k: None)
    out = d.run_defects4j_tests(b.project_name, b.bug_index, b.agent)
    assert out == "some other error"


def test_execute_get_info_success(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    info = "Root cause: broke\n" + "-" * 80 + "\n"
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=0, stdout=info))
    monkeypatch.setattr(d, "get_edited_files", lambda *a, **k: ["F.java"])
    monkeypatch.setattr(d, "extract_lines_range", lambda *a, **k: [(1, 5)])
    monkeypatch.setattr(d, "get_localization", lambda *a, **k: "LOCALIZATION")
    out = d.execute_get_info(b.project_name, b.bug_index, b.agent)
    assert "Root cause" in out
    assert "LOCALIZATION" in out


def test_execute_get_info_error(buggy_project_dir, monkeypatch):
    b = buggy_project_dir
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=1, stderr="nope"))
    out = d.execute_get_info(b.project_name, b.bug_index, b.agent)
    assert out == "Error: nope"


def test_get_info_delegates(buggy_project_dir, monkeypatch):
    monkeypatch.setattr(d, "execute_get_info", lambda *a, **k: "INFO")
    b = buggy_project_dir
    assert d.get_info(b.project_name, b.bug_index, b.agent) == "INFO"


# ===========================================================================
# prepare_command (pure string builder)
# ===========================================================================

def test_prepare_command_contains_expected_tokens():
    cmd = d.prepare_command("ws", "lang_1_buggy")
    assert "lspeclipse" in cmd
    assert "org.eclipse.jdt.ls.core.id1" in cmd
    assert "lsp_output.txt" in cmd
    # path joins workspace + project dir
    assert os.path.join("ws", "lang_1_buggy") in cmd


# ===========================================================================
# prepare_init_file (reads a template json, writes init file)
# ===========================================================================

def test_prepare_init_file_writes_filled_template(tmp_path, monkeypatch):
    template = {
        "params": {
            "rootPath": "",
            "rootUri": "",
            "initializationOptions": {"workspaceFolders": []},
            "workspaceFolders": [{"uri": "", "name": ""}],
        }
    }
    (tmp_path / "lsp_init_template.json").write_text(json.dumps(template))
    workspace = "ws"
    project_dir = "lang_1_buggy"
    os.makedirs(os.path.join(str(tmp_path), workspace, project_dir, "lspeclipse"))
    monkeypatch.chdir(tmp_path)
    d.prepare_init_file("/root", "file:///root", "file:///root/ws", workspace, project_dir)
    out_path = os.path.join(str(tmp_path), workspace, project_dir, "lspeclipse", "lsp_init_file.json")
    written = json.load(open(out_path))
    assert written["params"]["rootPath"] == "/root"
    assert written["params"]["rootUri"] == "file:///root"
    assert written["params"]["initializationOptions"]["workspaceFolders"] == ["file:///root/ws"]
    assert written["params"]["workspaceFolders"][0]["name"] == project_dir


# ===========================================================================
# prepare_lsp_env (subprocess.run mocked)
# ===========================================================================

def test_prepare_lsp_env_invokes_cp(monkeypatch):
    rec = []
    _patch_subprocess_run(monkeypatch, _FakeProc(returncode=0), recorder=rec)
    d.prepare_lsp_env("Lang", 1, "ws")
    assert rec, "subprocess.run should have been called"
    cmd = rec[0][0][0]
    assert cmd[0] == "cp"
    assert cmd[1] == "-r"
    assert cmd[2] == "lspeclipse"
    assert "lang_1_buggy" in cmd[3]


def test_prepare_lsp_env_handles_failure(monkeypatch):
    import subprocess as real_sub

    def boom(*a, **k):
        raise real_sub.CalledProcessError(1, "cp")

    monkeypatch.setattr(d.subprocess, "run", boom)
    # should swallow the error and not raise
    d.prepare_lsp_env("Lang", 1, "ws")


# ===========================================================================
# execute_command (Popen mocked)
# ===========================================================================

def test_execute_command_writes_requests(monkeypatch):
    recorded = {"writes": [], "killed": False, "flushed": 0}

    class FakeStdin:
        def write(self, data):
            recorded["writes"].append(data)

        def flush(self):
            recorded["flushed"] += 1

    class FakeProc:
        def __init__(self):
            self.stdin = FakeStdin()

        def kill(self):
            recorded["killed"] = True

    monkeypatch.setattr(d.subprocess, "Popen", lambda *a, **k: FakeProc())
    # avoid the real sleeps
    monkeypatch.setattr(d.time, "sleep", lambda *_a, **_k: None)

    d.execute_command("cmd", "INIT", "REQ")
    assert recorded["killed"] is True
    # both init and the real request go through the pipe, framed with Content-Length
    joined = "".join(recorded["writes"])
    assert "Content-Length:" in joined
    assert "INIT" in joined
    assert "REQ" in joined


# ===========================================================================
# lsp_hover (orchestration; mock nested IO/subprocess)
# ===========================================================================

def test_lsp_hover_returns_output(tmp_path, monkeypatch):
    project_dir = "lang_1_buggy"
    workspace = "ws"
    base = tmp_path
    # Pre-create the lspeclipse dir + init file + output so no env-prep needed
    lspdir = base / workspace / project_dir / "lspeclipse"
    lspdir.mkdir(parents=True)
    (lspdir / "lsp_init_file.json").write_text("{}")
    (base / workspace / project_dir / "lsp_output.txt").write_text("HOVER RESULT CONTENT")
    monkeypatch.chdir(base)
    monkeypatch.setattr(d, "prepare_command", lambda *a, **k: "cmd")
    monkeypatch.setattr(d, "execute_command", lambda *a, **k: None)
    agent = SimpleNamespace(config=SimpleNamespace(workspace=workspace))
    out = d.lsp_hover("Lang", 1, "src/Foo.java", 3, 2, agent)
    assert out == "HOVER RESULT CONTENT"


def test_lsp_hover_no_output_file(tmp_path, monkeypatch):
    project_dir = "lang_1_buggy"
    workspace = "ws"
    base = tmp_path
    lspdir = base / workspace / project_dir / "lspeclipse"
    lspdir.mkdir(parents=True)
    (lspdir / "lsp_init_file.json").write_text("{}")
    # no lsp_output.txt
    monkeypatch.chdir(base)
    monkeypatch.setattr(d, "prepare_command", lambda *a, **k: "cmd")
    monkeypatch.setattr(d, "execute_command", lambda *a, **k: None)
    agent = SimpleNamespace(config=SimpleNamespace(workspace=workspace))
    out = d.lsp_hover("Lang", 1, "src/Foo.java", 3, 2, agent)
    assert "ERROR NO OUTPUT" in out


def test_lsp_hover_empty_output(tmp_path, monkeypatch):
    project_dir = "lang_1_buggy"
    workspace = "ws"
    base = tmp_path
    lspdir = base / workspace / project_dir / "lspeclipse"
    lspdir.mkdir(parents=True)
    (lspdir / "lsp_init_file.json").write_text("{}")
    (base / workspace / project_dir / "lsp_output.txt").write_text("")
    monkeypatch.chdir(base)
    monkeypatch.setattr(d, "prepare_command", lambda *a, **k: "cmd")
    monkeypatch.setattr(d, "execute_command", lambda *a, **k: None)
    agent = SimpleNamespace(config=SimpleNamespace(workspace=workspace))
    out = d.lsp_hover("Lang", 1, "src/Foo.java", 3, 2, agent)
    assert "ERROR EMPTY OUTPUT" in out


# ===========================================================================
# ask_chatgpt (LLM mocked)
# ===========================================================================

class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeChat:
    last_messages = None

    def __init__(self, *a, **k):
        pass

    def invoke(self, messages):
        _FakeChat.last_messages = messages
        return _FakeResponse("ANSWER")


def test_ask_chatgpt_first_call(monkeypatch):
    monkeypatch.setattr(d, "ChatOpenAI", _FakeChat)
    agent = SimpleNamespace(
        config=SimpleNamespace(static_llm="gpt-4", temperature=0.0),
        ask_chatgpt=None,
    )
    out = d.ask_chatgpt("why bug?", agent)
    assert out == "ANSWER"
    # conversation history recorded (system + human + response)
    assert agent.ask_chatgpt is not None
    assert len(agent.ask_chatgpt) >= 2


def test_ask_chatgpt_followup(monkeypatch):
    monkeypatch.setattr(d, "ChatOpenAI", _FakeChat)
    prior = [d.SystemMessage(content="sys"), d.HumanMessage(content="prev")]
    agent = SimpleNamespace(
        config=SimpleNamespace(static_llm="gpt-4", temperature=0.0),
        ask_chatgpt=prior,
    )
    out = d.ask_chatgpt("follow up", agent)
    assert out == "ANSWER"
    # appended the new human message and the response
    assert any(getattr(m, "content", "") == "follow up" for m in agent.ask_chatgpt)


# ===========================================================================
# validate_fix_against_hypothesis (LLM mocked, temperature branching)
# ===========================================================================

def test_validate_fix_against_hypothesis_default_temp(monkeypatch):
    captured = {}

    class Chat:
        def __init__(self, *a, **k):
            captured.update(k)

        def invoke(self, messages):
            return _FakeResponse("VALIDATION")

    monkeypatch.setattr(d, "ChatOpenAI", Chat)
    monkeypatch.delenv("TEMPERATURE", raising=False)
    out = d.validate_fix_against_hypothesis("bug", "hyp", "fix", "gpt-4")
    assert out == "VALIDATION"
    assert captured["temperature"] == 0.0


def test_validate_fix_against_hypothesis_gpt5_forces_temp_1(monkeypatch):
    captured = {}

    class Chat:
        def __init__(self, *a, **k):
            captured.update(k)

        def invoke(self, messages):
            return _FakeResponse("VALIDATION")

    monkeypatch.setattr(d, "ChatOpenAI", Chat)
    out = d.validate_fix_against_hypothesis("bug", "hyp", "fix", "gpt-5-mini")
    assert captured["temperature"] == 1.0


# ===========================================================================
# extract_method_code (antlr) and extract_similar_functions_calls
# ===========================================================================

def test_extract_method_code_returns_body(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "Foo.java").write_text(
        "public class Foo {\n"
        "    public int bar(int x) {\n"
        "        return x + 1;\n"
        "    }\n"
        "}\n"
    )
    out = d.extract_method_code(b.project_name, b.bug_index, "Foo.java", "bar", b.agent)
    assert "Implementation candidate 0" in out
    assert "return x + 1;" in out


def test_extract_method_code_no_match(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "Foo.java").write_text(
        "public class Foo { public int bar(int x) { return x; } }\n"
    )
    out = d.extract_method_code(b.project_name, b.bug_index, "Foo.java", "nope", b.agent)
    # header present, but no candidates
    assert "method name nope" in out
    assert "Implementation candidate" not in out


def test_extract_similar_functions_calls_found(buggy_project_dir):
    b = buggy_project_dir
    (b.project_dir / "Foo.java").write_text(
        "public class Foo {\n"
        "    void m() { compute(1); compute(2, 3); }\n"
        "}\n"
    )
    out = d.extract_similar_functions_calls(
        b.project_name, b.bug_index, "Foo.java", "compute(1)", b.agent
    )
    assert "similar calls were found" in out
    assert "compute(2, 3)" in out


def test_extract_similar_functions_calls_no_calls(buggy_project_dir, quiet_info_logging):
    b = buggy_project_dir
    (b.project_dir / "Foo.java").write_text("public class Foo { int x = 1; }\n")
    out = d.extract_similar_functions_calls(
        b.project_name, b.bug_index, "Foo.java", "int y = 2;", b.agent
    )
    assert "No function calls were found" in out


# ===========================================================================
# extract_function_def_context (uses extract_method_code + tiktoken)
# ===========================================================================

def test_extract_function_def_context_returns_preceding(monkeypatch, tmp_path):
    # workspace is hardcoded to ./auto_gpt_workspace, so build tree there
    ws = tmp_path / "auto_gpt_workspace"
    pdir = ws / "lang_1_buggy"
    pdir.mkdir(parents=True)
    code = "HEADER STUFF\npublic int bar(){return 1;}\n"
    (pdir / "Foo.java").write_text(code)
    monkeypatch.chdir(tmp_path)
    agent = SimpleNamespace(config=SimpleNamespace(workspace_path=str(ws)))
    # preprocess_paths resolves "Foo.java" directly (file exists at project_dir)
    monkeypatch.setattr(d, "preprocess_paths", lambda agent, pn, bi, fp: "Foo.java")
    # extract_method_code returns a STRING; context uses extracted[0] = first char.
    # Make it return a substring present in the file so .find succeeds.
    monkeypatch.setattr(d, "extract_method_code", lambda *a, **k: "HEADER STUFF\npublic")
    out = d.extract_function_def_context("Lang", 1, "bar", "Foo.java", agent)
    # context is the slice before the found method_body; method_body starts at 0
    assert out == ""


def test_extract_function_def_context_method_not_found_raises(monkeypatch, tmp_path):
    ws = tmp_path / "auto_gpt_workspace"
    pdir = ws / "lang_1_buggy"
    pdir.mkdir(parents=True)
    (pdir / "Foo.java").write_text("some content\n")
    monkeypatch.chdir(tmp_path)
    agent = SimpleNamespace(config=SimpleNamespace(workspace_path=str(ws)))
    monkeypatch.setattr(d, "preprocess_paths", lambda *a, **k: "Foo.java")
    monkeypatch.setattr(d, "extract_method_code", lambda *a, **k: "ZZZ not in file")
    with pytest.raises(ValueError):
        d.extract_function_def_context("Lang", 1, "bar", "Foo.java", agent)


# ===========================================================================
# auto_complete_functions (LLM mocked, temperature branching)
# ===========================================================================

def test_auto_complete_functions_invokes_llm(monkeypatch):
    captured = {}

    class Chat:
        def __init__(self, *a, **k):
            captured.update(k)

        def invoke(self, messages):
            return _FakeResponse("GENERATED CODE")

    monkeypatch.setattr(d, "ChatOpenAI", Chat)
    monkeypatch.setattr(d, "extract_function_def_context", lambda *a, **k: "context")
    agent = SimpleNamespace(config=SimpleNamespace(static_llm="gpt-4", temperature=0.3))
    out = d.auto_complete_functions("Lang", 1, "Foo.java", "bar", agent)
    assert out == "GENERATED CODE"
    assert captured["temperature"] == 0.3


def test_auto_complete_functions_gpt5_temp(monkeypatch):
    captured = {}

    class Chat:
        def __init__(self, *a, **k):
            captured.update(k)

        def invoke(self, messages):
            return _FakeResponse("X")

    monkeypatch.setattr(d, "ChatOpenAI", Chat)
    monkeypatch.setattr(d, "extract_function_def_context", lambda *a, **k: "ctx")
    agent = SimpleNamespace(config=SimpleNamespace(static_llm="gpt-5", temperature=0.3))
    d.auto_complete_functions("Lang", 1, "Foo.java", "bar", agent)
    assert captured["temperature"] == 1.0


# ===========================================================================
# extract_test_code
# ===========================================================================

def test_extract_test_code_no_failing_test(buggy_project_dir):
    b = buggy_project_dir
    # the *_test.txt is read from workspace/<project_dir>_test.txt
    (b.workspace / "lang_1_buggy_test.txt").write_text("no failing pattern here")
    out = d.extract_test_code(b.project_name, b.bug_index, "FooTest.java", b.agent)
    assert "No test function found" in out


def test_extract_test_code_extracts_function(buggy_project_dir):
    b = buggy_project_dir
    test_msg = "Failing tests: 1\n  - org.foo.FooTest::testBar\n"
    (b.workspace / "lang_1_buggy_test.txt").write_text(test_msg)
    # the test file path must exist directly under project dir
    (b.project_dir / "FooTest.java").write_text(
        "public class FooTest {\n"
        "    public void testBar() { assertTrue(true); }\n"
        "    public void testOther() {}\n"
        "}\n"
    )
    out = d.extract_test_code(b.project_name, b.bug_index, "FooTest.java", b.agent)
    assert "testBar" in out
    assert "assertTrue(true)" in out


def test_extract_test_code_function_not_in_file(buggy_project_dir):
    b = buggy_project_dir
    test_msg = "Failing tests: 1\n  - org.foo.FooTest::missingFunc\n"
    (b.workspace / "lang_1_buggy_test.txt").write_text(test_msg)
    (b.project_dir / "FooTest.java").write_text(
        "public class FooTest { public void testBar() {} }\n"
    )
    out = d.extract_test_code(b.project_name, b.bug_index, "FooTest.java", b.agent)
    # function "missingFunc" not found -> 'public void missingFunc' index == -1
    assert out is None
