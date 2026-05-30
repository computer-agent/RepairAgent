"""Unit tests for pure parsing/utility logic in the defects4j command modules.

Targets functions that are genuinely unit-testable without Java, the defects4j
toolchain, or a fully-built Agent:

* ``autogpt.commands.defects4j.extract_file_name`` / ``get_edited_files``
* ``autogpt.commands.defects4j.parse_buggy_lines``
* ``autogpt.commands.defects4j.create_fix_template``
* ``autogpt.commands.defects4j.extract_targeted_lines``
* ``autogpt.commands.defects4j.remove_comments``
* ``autogpt.commands.defects4j.extract_failing_test`` / ``extract_root_cause``
* ``create_files_index.list_java_files`` (also mirrored in the static module)

File-reading helpers resolve their inputs through paths relative to the current
working directory (``defects4j/...``), so those tests ``monkeypatch.chdir`` into
a tmp tree rather than touching the real repo data.
"""

import os

import pytest

import create_files_index
import tests.conftest  # noqa: F401  (ensures sys.path / chdir / stubs are in place)
from autogpt.commands import defects4j

# --------------------------------------------------------------------------
# extract_file_name : parse a single 'diff --git' line
# --------------------------------------------------------------------------


def test_extract_file_name_strips_leading_segment():
    line = "diff --git a/src/main/java/Foo.java b/src/main/java/Foo.java"
    # third whitespace token is "a/src/main/java/Foo.java"; the leading
    # path segment ("a") is dropped and the rest re-joined with "/".
    assert defects4j.extract_file_name(line) == "src/main/java/Foo.java"


def test_extract_file_name_single_segment():
    line = "diff --git a/Foo.java b/Foo.java"
    assert defects4j.extract_file_name(line) == "Foo.java"


# --------------------------------------------------------------------------
# get_edited_files : collect file names from a .src.patch
# --------------------------------------------------------------------------


def _make_patch(base_dir, name, index, content):
    patch_dir = os.path.join(
        base_dir, "defects4j", "framework", "projects", name, "patches"
    )
    os.makedirs(patch_dir, exist_ok=True)
    path = os.path.join(patch_dir, "{}.src.patch".format(index))
    with open(path, "w") as fh:
        fh.write(content)
    return path


def test_get_edited_files_multiple_files(tmp_path, monkeypatch):
    content = (
        "diff --git a/src/Foo.java b/src/Foo.java\n"
        "index 111..222 100644\n"
        "--- a/src/Foo.java\n"
        "+++ b/src/Foo.java\n"
        "@@ -1 +1 @@\n"
        "+x\n"
        "diff --git a/src/Bar.java b/src/Bar.java\n"
        "+y\n"
    )
    _make_patch(str(tmp_path), "Lang", "1", content)
    monkeypatch.chdir(tmp_path)
    assert defects4j.get_edited_files("Lang", "1") == ["src/Foo.java", "src/Bar.java"]


def test_get_edited_files_empty_patch(tmp_path, monkeypatch):
    _make_patch(str(tmp_path), "Lang", "2", "no diff headers here\njust text\n")
    monkeypatch.chdir(tmp_path)
    assert defects4j.get_edited_files("Lang", "2") == []


# --------------------------------------------------------------------------
# parse_buggy_lines : "path#lineno#code" -> {path: [(lineno, code), ...]}
# --------------------------------------------------------------------------


def test_parse_buggy_lines_groups_by_file():
    lines = [
        "src/Foo.java#10#  int x;",
        "src/Foo.java#12#  int y;",
        "src/Bar.java#5#code",
    ]
    parsed = defects4j.parse_buggy_lines(lines)
    assert parsed == {
        "src/Foo.java": [("10", "  int x;"), ("12", "  int y;")],
        "src/Bar.java": [("5", "code")],
    }


def test_parse_buggy_lines_empty_input():
    assert defects4j.parse_buggy_lines([]) == {}


def test_parse_buggy_lines_extra_hash_in_code_is_preserved():
    # Code containing '#' beyond the first two separators is rejoined and kept
    # intact (no longer truncated at the third token).
    parsed = defects4j.parse_buggy_lines(["src/Foo.java#10#a # b"])
    assert parsed == {"src/Foo.java": [("10", "a # b")]}


def test_parse_buggy_lines_malformed_line_skipped():
    # A line without the expected two '#' separators is skipped rather than
    # raising IndexError; well-formed lines in the same batch still parse.
    parsed = defects4j.parse_buggy_lines(
        ["this-line-has-no-hash", "src/Foo.java#10#x = 1"]
    )
    assert parsed == {"src/Foo.java": [("10", "x = 1")]}


# --------------------------------------------------------------------------
# create_fix_template : reads a .buggy.lines file and builds a template string
# --------------------------------------------------------------------------


def _make_buggy_lines(base_dir, name, index, content):
    d = os.path.join(base_dir, "defects4j", "buggy-lines")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "{}-{}.buggy.lines".format(name, index))
    with open(path, "w") as fh:
        fh.write(content)
    return path


def test_create_fix_template_contains_expected_fields(tmp_path, monkeypatch):
    _make_buggy_lines(
        str(tmp_path),
        "Lang",
        "1",
        "src/Foo.java#10#  int x;\nsrc/Foo.java#12#  int y;\n",
    )
    monkeypatch.chdir(tmp_path)
    out = defects4j.create_fix_template("Lang", "1")

    assert isinstance(out, str)
    assert '"file_name": "src/Foo.java"' in out
    # target_lines preserves the (lineno, code) pairs as nested lists in JSON
    assert '["10", "  int x;"]' in out
    # The three edit buckets are annotated with guidance comments.
    assert "here put the list of modification" in out
    assert "here put the lines number to delete" in out
    assert "here put the list of insertion" in out


def test_create_fix_template_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        defects4j.create_fix_template("DoesNotExist", "99")


# --------------------------------------------------------------------------
# extract_targeted_lines : flatten edit dicts into a list of ints
# --------------------------------------------------------------------------


def test_extract_targeted_lines_all_edit_types():
    changes = [
        {
            "deletions": ["3"],
            "modifications": [{"line_number": "7"}],
            "insertions": [{"line_number": 9}],
        }
    ]
    # order: deletions, then modifications, then insertions; all coerced to int
    assert defects4j.extract_targeted_lines(changes) == [3, 7, 9]


def test_extract_targeted_lines_missing_keys_and_multiple_dicts():
    changes = [
        {"deletions": ["1"]},  # no modifications/insertions keys -> defaults []
        {"modifications": [{"line_number": 2}, {"line_number": "4"}]},
    ]
    assert defects4j.extract_targeted_lines(changes) == [1, 2, 4]


def test_extract_targeted_lines_empty():
    assert defects4j.extract_targeted_lines([]) == []


# --------------------------------------------------------------------------
# remove_comments : strip // and /* */ comments from Java source
# --------------------------------------------------------------------------


def test_remove_comments_strips_both_styles():
    code = "int x; // trailing\n/* block\ncomment */int y;"
    out = defects4j.remove_comments(code)
    assert "trailing" not in out
    assert "block" not in out
    assert "int x;" in out
    assert "int y;" in out


def test_remove_comments_no_comments_unchanged():
    code = "int x;\nint y;"
    assert defects4j.remove_comments(code) == code


# --------------------------------------------------------------------------
# extract_failing_test : parse the "Failing tests:" block
# --------------------------------------------------------------------------


def test_extract_failing_test_match():
    msg = "header\nFailing tests: 2\n  - org.foo.BarTest::testBaz\nfooter"
    assert defects4j.extract_failing_test(msg) == {
        "num_failures": 2,
        "class_name": "org.foo.BarTest",
        "function_name": "testBaz",
    }


def test_extract_failing_test_no_match_returns_none():
    assert defects4j.extract_failing_test("nothing relevant here") is None


# --------------------------------------------------------------------------
# extract_root_cause : slice text between "Root cause" and the separator
# --------------------------------------------------------------------------


def test_extract_root_cause_extracts_segment():
    sep = "-" * 80
    info = "intro\nRoot cause: the thing broke\nmore details\n" + sep + "\ntail"
    out = defects4j.extract_root_cause(info)
    assert out.startswith("Root cause")
    assert "the thing broke" in out
    assert sep not in out


def test_extract_root_cause_keyword_absent_returns_empty():
    # find("Root cause") == -1 -> start_cause = -1; the slice from -1 up to the
    # separator-relative offset collapses to an empty string. Documents the
    # (arguably surprising) behavior when the keyword is missing.
    sep = "-" * 80
    assert defects4j.extract_root_cause("no keyword here " + sep) == ""


# --------------------------------------------------------------------------
# list_java_files : recursively collect .java files
# --------------------------------------------------------------------------


def test_list_java_files_only_java_and_recurses(tmp_path):
    (tmp_path / "A.java").write_text("class A {}")
    (tmp_path / "readme.txt").write_text("not java")
    (tmp_path / "B.py").write_text("print()")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "C.java").write_text("class C {}")

    result = create_files_index.list_java_files(str(tmp_path))

    # Only .java files are returned, recursion into sub/ works.
    java_basenames = sorted(os.path.basename(p) for p in result)
    assert java_basenames == ["A.java", "C.java"]
    # Nested files are made relative to main_dir (prefix "<main_dir>/" stripped).
    assert any(p == os.path.join("sub", "C.java") for p in result)


def test_list_java_files_empty_dir(tmp_path):
    assert create_files_index.list_java_files(str(tmp_path)) == []


def test_list_java_files_top_level_path_relativized(tmp_path):
    # A file directly under main_dir is returned relative to main_dir (just its
    # name), consistent with nested files.
    (tmp_path / "Top.java").write_text("class Top {}")
    result = create_files_index.list_java_files(str(tmp_path))
    assert len(result) == 1
    assert result[0] == "Top.java"
