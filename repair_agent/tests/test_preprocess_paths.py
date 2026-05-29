"""Tests for autogpt.commands.defects4j.preprocess_paths (file-path resolution).

The function turns dotted/.java logical paths into real on-disk relative paths
inside ``<workspace>/<project>_<index>_buggy``. When the direct path does not
exist it builds/reads ``files_index.txt`` and tries to disambiguate using a
substring match (and, when several match, an exact path-suffix match).
"""

import os

import pytest

from autogpt.commands.defects4j import preprocess_paths


def _call(bp, filepath):
    return preprocess_paths(bp.agent, bp.project_name, bp.bug_index, filepath)


def test_existing_path_returned_as_is(buggy_project_dir):
    # No dots in the path; the file exists directly under project_dir, so the
    # dot->slash logic is a no-op and the path is returned unchanged.
    bp = buggy_project_dir
    bp.write("Foo", "content")
    assert _call(bp, "Foo") == "Foo"


def test_existing_nested_path_with_slashes(buggy_project_dir):
    # A path that already exists with slashes (no dots) is returned unchanged.
    bp = buggy_project_dir
    bp.write("src/main/Bar", "content")
    assert _call(bp, "src/main/Bar") == "src/main/Bar"


def test_java_suffix_dots_become_slashes_and_resolves(buggy_project_dir):
    # 'a.b.Foo.java' -> 'a/b/Foo.java'; we create that exact file so it exists.
    bp = buggy_project_dir
    bp.write("a/b/Foo.java", "class Foo {}")
    assert _call(bp, "a.b.Foo.java") == "a/b/Foo.java"


def test_non_java_input_dots_become_slashes(buggy_project_dir):
    # Non-.java input: every dot becomes a slash. Create matching file so it
    # resolves to the dotted->slash form directly (the existence branch).
    bp = buggy_project_dir
    bp.write("com/example/pkg", "x")
    assert _call(bp, "com.example.pkg") == "com/example/pkg"


def test_files_index_single_candidate_substring_match(buggy_project_dir):
    # The requested path does not exist directly, but exactly one indexed java
    # file contains it as a substring -> resolve to that candidate.
    bp = buggy_project_dir
    bp.write("src/main/java/org/foo/Widget.java", "class Widget {}")
    bp.write("src/test/java/org/foo/Unrelated.java", "class Unrelated {}")
    # 'Widget.java' is not directly present at project_dir root, triggering the
    # index lookup; only one indexed file contains 'Widget.java'.
    result = _call(bp, "Widget.java")
    assert result == "src/main/java/org/foo/Widget.java"
    # files_index.txt is created on first miss.
    assert os.path.exists(os.path.join(str(bp.project_dir), "files_index.txt"))


def test_ambiguous_no_exact_suffix_returns_message_not_exception(buggy_project_dir):
    # Two files both contain the substring 'Foo' but neither is an exact suffix
    # match for the (slash-converted) query, so the result is the ambiguous
    # message string (the recent fix: must NOT raise).
    bp = buggy_project_dir
    bp.write("src/a/FooBar.java", "class FooBar {}")
    bp.write("src/b/FooBaz.java", "class FooBaz {}")
    # Query 'Foo' (no .java): becomes 'Foo'; substring-matches both, and neither
    # equals 'Foo' nor ends with '/Foo'.
    result = _call(bp, "Foo")
    assert isinstance(result, str)
    assert result.startswith("The filepath")
    assert "ambiguous" in result


def test_ambiguous_with_single_exact_suffix_resolves(buggy_project_dir):
    # Multiple substring candidates, but exactly one is an exact '/<filepath>'
    # suffix match -> resolve to that one.
    bp = buggy_project_dir
    bp.write("src/main/java/org/foo/Helper.java", "class Helper {}")
    bp.write("src/main/java/org/foo/HelperFactory.java", "class HelperFactory {}")
    # 'Helper.java' is a substring of both, but only the first ends with
    # '/Helper.java'.
    result = _call(bp, "Helper.java")
    assert result == "src/main/java/org/foo/Helper.java"


def test_missing_file_returns_does_not_exist_message(buggy_project_dir):
    # Nothing matches the substring -> 'does not exist' message. Note the
    # message uses the slash-converted filepath.
    bp = buggy_project_dir
    bp.write("src/main/Existing.java", "class Existing {}")
    result = _call(bp, "Nope.java")
    assert result == "The filepath Nope.java does not exist."


def test_missing_file_message_uses_slash_converted_path(buggy_project_dir):
    # For a dotted non-existent path, the message reflects the slash-converted
    # value, confirming the conversion happens before the lookup.
    bp = buggy_project_dir
    bp.write("src/main/Existing.java", "class Existing {}")
    result = _call(bp, "no.such.Thing.java")
    assert result == "The filepath no/such/Thing.java does not exist."


def test_files_index_reused_when_already_present(buggy_project_dir):
    # If files_index.txt already exists, it is read (not rebuilt). We seed a
    # stale index pointing at a path that maps to a real file to confirm the
    # function trusts the existing index content.
    bp = buggy_project_dir
    # Create the real file so the resolved candidate also exists on disk.
    bp.write("custom/path/Seeded.java", "class Seeded {}")
    index_path = os.path.join(str(bp.project_dir), "files_index.txt")
    with open(index_path, "w") as f:
        f.write("custom/path/Seeded.java\n")
    result = _call(bp, "Seeded.java")
    assert result == "custom/path/Seeded.java"


def test_dot_conversion_then_index_match(buggy_project_dir):
    # A dotted .java logical class name that doesn't exist as a literal path,
    # but whose slash form is a suffix of an indexed file.
    bp = buggy_project_dir
    bp.write("src/main/java/org/apache/commons/Lang.java", "class Lang {}")
    # 'org.apache.commons.Lang.java' -> 'org/apache/commons/Lang.java',
    # substring of the indexed path; single candidate resolves it.
    result = _call(bp, "org.apache.commons.Lang.java")
    assert result == "src/main/java/org/apache/commons/Lang.java"
