"""Shared pytest scaffolding for the RepairAgent test suite.

Responsibilities:
1. Put the ``repair_agent`` directory on ``sys.path`` so ``import autogpt.*`` and
   top-level modules (``preprocess_paths``, ``create_files_index``, …) resolve.
2. Run from the ``repair_agent`` directory so the project's relative file reads
   (e.g. ``open("commands_interface.json")``) work during tests.
3. Stub the heavy/optional ``langchain`` stack in ``sys.modules`` BEFORE any
   project code is imported. Latest ``langchain`` requires pydantic v2, which
   conflicts with the project's pydantic-v1 config; the LLM call paths are mocked
   in tests, so a lightweight stub is all that's needed for imports to succeed.

These run at import time (conftest is imported before test collection).
"""

import os
import sys
import types

REPAIR_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 1. import path ---------------------------------------------------------
if REPAIR_AGENT_DIR not in sys.path:
    sys.path.insert(0, REPAIR_AGENT_DIR)

# --- 2. working directory ---------------------------------------------------
os.chdir(REPAIR_AGENT_DIR)


# --- 3. langchain stubs -----------------------------------------------------
def _stub(name, **attrs):
    """Register a stub module under ``name`` unless the real one is importable."""
    if name in sys.modules:
        return sys.modules[name]
    try:  # prefer the real package when it happens to be installed
        __import__(name)
        return sys.modules[name]
    except Exception:
        pass
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _StubMessage:  # stand-in for langchain message classes
    def __init__(self, content="", *args, **kwargs):
        self.content = content


class _StubChat:  # stand-in for ChatOpenAI / ChatAnthropic
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def invoke(self, *args, **kwargs):  # pragma: no cover - tests mock this
        raise RuntimeError(
            "langchain is stubbed in tests; mock the LLM call explicitly instead."
        )


_stub("langchain")
_stub("langchain.chat_models", ChatOpenAI=_StubChat)
_stub("langchain.schema")
_stub(
    "langchain.schema.messages",
    HumanMessage=_StubMessage,
    SystemMessage=_StubMessage,
    AIMessage=_StubMessage,
)
_stub("langchain_anthropic", ChatAnthropic=_StubChat)


# --- shared fixtures --------------------------------------------------------
import pytest  # noqa: E402  (import after sys.path setup)


@pytest.fixture
def repair_root():
    """Absolute path to the repair_agent directory."""
    return REPAIR_AGENT_DIR


@pytest.fixture
def make_fake_agent():
    """Factory for a minimal stand-in agent with a ``config.workspace_path``.

    Many command helpers only touch ``agent.config.workspace_path``; this avoids
    constructing a real (heavy) Agent. Pass extra attributes via ``config_attrs``
    or ``agent_attrs``.
    """
    from types import SimpleNamespace

    def _make(workspace_path, config_attrs=None, agent_attrs=None):
        config = SimpleNamespace(workspace_path=str(workspace_path))
        for key, value in (config_attrs or {}).items():
            setattr(config, key, value)
        agent = SimpleNamespace(config=config)
        for key, value in (agent_attrs or {}).items():
            setattr(agent, key, value)
        return agent

    return _make


@pytest.fixture
def buggy_project_dir(tmp_path, make_fake_agent):
    """Create a fake ``<workspace>/<project>_<index>_buggy`` tree with Java files.

    Returns a SimpleNamespace with: ``workspace``, ``agent``, ``project_name``,
    ``bug_index``, ``project_dir`` (absolute), and a ``write(rel, text)`` helper.
    """
    from types import SimpleNamespace

    workspace = tmp_path / "auto_gpt_workspace"
    project_name, bug_index = "Lang", 1
    project_dir = workspace / f"{project_name.lower()}_{bug_index}_buggy"
    project_dir.mkdir(parents=True)

    def write(rel, text):
        path = project_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    return SimpleNamespace(
        workspace=workspace,
        agent=make_fake_agent(workspace),
        project_name=project_name,
        bug_index=bug_index,
        project_dir=project_dir,
        write=write,
    )
