"""Scaffolding smoke tests: confirm the suite can import the project's modules.

If these fail, the conftest path/stub setup is broken and every other test in the
suite is unreliable — fix this first.
"""

import importlib

import pytest

# Modules that must import cleanly (real modules, with langchain stubbed).
IMPORTABLE_MODULES = [
    "preprocess_paths",
    "create_files_index",
    "autogpt.json_utils.utilities",
    "autogpt.config.config",
    "autogpt.llm.utils",
    "autogpt.commands.defects4j",
    "autogpt.commands.defects4j_static",
    "autogpt.agents.agent",
    "autogpt.agents.base",
]


@pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None


def test_fake_agent_fixture(make_fake_agent, tmp_path):
    agent = make_fake_agent(tmp_path)
    assert agent.config.workspace_path == str(tmp_path)


def test_buggy_project_dir_fixture(buggy_project_dir):
    p = buggy_project_dir.write("src/main/java/A.java", "class A {}\n")
    assert p.exists()
    assert buggy_project_dir.project_dir.name == "lang_1_buggy"
