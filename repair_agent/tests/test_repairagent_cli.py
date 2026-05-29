"""Tests for repairagent.py — the setup wizard helpers and the click CLI.

All subprocess / prompt / filesystem-touching behavior is mocked. Nothing real
(apt, pip, git, docker, java, defects4j, the real experimental_setups dir) is
executed or written; we monkeypatch ``repairagent.subprocess.run``,
``repairagent.SCRIPT_DIR`` and the rich prompt classes as needed.
"""
import subprocess
import types

import pytest
from click.testing import CliRunner

import repairagent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _completed(returncode=0, stdout="", stderr=""):
    """Build a fake subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


# ===========================================================================
# 1. PURE PARSERS
# ===========================================================================

class TestParseBugsString:
    def test_basic(self):
        assert repairagent.parse_bugs_string("Chart 1,Math 5") == [
            ("Chart", "1"), ("Math", "5")
        ]

    def test_with_spaces_around_commas(self):
        assert repairagent.parse_bugs_string("Chart 1, Math 5") == [
            ("Chart", "1"), ("Math", "5")
        ]

    def test_surrounding_quotes_stripped(self):
        assert repairagent.parse_bugs_string('"Chart 1,Math 5"') == [
            ("Chart", "1"), ("Math", "5")
        ]
        assert repairagent.parse_bugs_string("'Lang 3'") == [("Lang", "3")]

    def test_empty_string(self):
        assert repairagent.parse_bugs_string("") == []

    def test_whitespace_only(self):
        assert repairagent.parse_bugs_string("   ") == []

    def test_invalid_entries_skipped(self):
        # single token, three tokens, and an empty entry between commas
        result = repairagent.parse_bugs_string("Chart,Math 5 6,,Lang 2")
        assert result == [("Lang", "2")]

    def test_per_entry_quotes_stripped(self):
        assert repairagent.parse_bugs_string('"Chart 1","Math 5"') == [
            ("Chart", "1"), ("Math", "5")
        ]


class TestLoadBugsFile:
    def test_parsing(self, tmp_path):
        f = tmp_path / "bugs.txt"
        f.write_text(
            "Chart 1\n"
            "\n"               # blank line skipped
            "Math 5 extra\n"   # extra tokens -> first two kept
            "   \n"            # whitespace-only skipped
            "Lang 7\n"
        )
        assert repairagent.load_bugs_file(str(f)) == [
            ("Chart", "1"), ("Math", "5"), ("Lang", "7")
        ]

    def test_single_token_line_skipped(self, tmp_path):
        f = tmp_path / "bugs.txt"
        f.write_text("Chart\nMath 2\n")
        assert repairagent.load_bugs_file(str(f)) == [("Math", "2")]

    def test_empty_file(self, tmp_path):
        f = tmp_path / "bugs.txt"
        f.write_text("")
        assert repairagent.load_bugs_file(str(f)) == []


# ===========================================================================
# 2. ENV / INSPECTION HELPERS
# ===========================================================================

class TestIsInContainer:
    def test_dockerenv(self, monkeypatch):
        monkeypatch.setattr(repairagent.os.path, "exists",
                            lambda p: p == "/.dockerenv")
        monkeypatch.delenv("CODESPACES", raising=False)
        assert repairagent.is_in_container() is True

    def test_codespaces(self, monkeypatch):
        monkeypatch.setattr(repairagent.os.path, "exists", lambda p: False)
        monkeypatch.setenv("CODESPACES", "true")
        assert repairagent.is_in_container() is True

    def test_run_containerenv(self, monkeypatch):
        monkeypatch.setattr(repairagent.os.path, "exists",
                            lambda p: p == "/run/.containerenv")
        monkeypatch.delenv("CODESPACES", raising=False)
        assert repairagent.is_in_container() is True

    def test_negative(self, monkeypatch):
        monkeypatch.setattr(repairagent.os.path, "exists", lambda p: False)
        monkeypatch.delenv("CODESPACES", raising=False)
        assert repairagent.is_in_container() is False


class TestHasApiKey:
    def test_env_var_set(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
        assert repairagent._has_api_key("OPENAI_API_KEY") is True

    def test_placeholder_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "GLOBAL-API-KEY-PLACEHOLDER")
        # point SCRIPT_DIR to an empty dir so no .env rescues it
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent._has_api_key("OPENAI_API_KEY") is False

    def test_unset_no_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent._has_api_key("OPENAI_API_KEY") is False

    def test_from_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "# a comment\nANTHROPIC_API_KEY=sk-ant-123\nFOO=bar\n"
        )
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent._has_api_key("ANTHROPIC_API_KEY") is True

    def test_env_file_placeholder_ignored(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "OPENAI_API_KEY=GLOBAL-API-KEY-PLACEHOLDER\n"
        )
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent._has_api_key("OPENAI_API_KEY") is False


class TestCheckPythonPackages:
    def test_all_present(self, monkeypatch):
        monkeypatch.setattr(repairagent, "__import__",
                            lambda name, *a, **k: types.ModuleType(name),
                            raising=False)
        # repairagent uses the builtin __import__ inside the function; patch builtins
        import builtins
        monkeypatch.setattr(builtins, "__import__",
                            lambda name, *a, **k: types.ModuleType(name))
        assert repairagent._check_python_packages() is True

    def test_one_missing(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "tiktoken":
                raise ImportError("no tiktoken")
            return types.ModuleType(name)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert repairagent._check_python_packages() is False


class TestCheckBootstrapDeps:
    def test_all_present(self, monkeypatch):
        import builtins
        monkeypatch.setattr(builtins, "__import__",
                            lambda name, *a, **k: types.ModuleType(name))
        assert repairagent._check_bootstrap_deps() == []

    def test_missing(self, monkeypatch):
        import builtins

        def fake_import(name, *a, **k):
            if name == "rich":
                raise ImportError("no rich")
            return types.ModuleType(name)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert repairagent._check_bootstrap_deps() == ["rich"]


class TestCheckCommand:
    def test_success_returns_stdout(self, monkeypatch):
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(0, "  output here  "))
        assert repairagent._check_command(["foo"]) == "output here"

    def test_nonzero_returns_none(self, monkeypatch):
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(1, "stuff"))
        assert repairagent._check_command(["foo"]) is None

    def test_file_not_found(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError()
        monkeypatch.setattr(repairagent.subprocess, "run", boom)
        assert repairagent._check_command(["nope"]) is None

    def test_timeout(self, monkeypatch):
        def boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="x", timeout=10)
        monkeypatch.setattr(repairagent.subprocess, "run", boom)
        assert repairagent._check_command(["slow"]) is None


class TestCheckEnvironment:
    def test_all_present(self, monkeypatch, tmp_path):
        # java succeeds (stderr output), every _check_command succeeds
        def fake_run(cmd, *a, **k):
            if cmd and cmd[0] == "java":
                return _completed(0, stderr='openjdk version "11.0"')
            return _completed(0, stdout="ok")
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        monkeypatch.setattr(repairagent, "_check_python_packages", lambda: True)
        monkeypatch.setattr(repairagent, "_has_api_key", lambda k: True)
        # defects4j path check: point SCRIPT_DIR somewhere & make d4j exist
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: "/usr/bin/" + x)

        checks = repairagent.check_environment()

        assert set(checks) >= {
            "Python 3.10+", "Java (JDK)", "Perl", "cpanminus",
            "Subversion", "Defects4J", "Python packages", "API key",
        }
        # each value is a (bool, str) tuple
        for ok, detail in checks.values():
            assert isinstance(ok, bool)
            assert isinstance(detail, str)
        assert checks["Java (JDK)"][0] is True
        assert checks["Perl"][0] is True
        assert checks["Python packages"] == (True, "installed")
        assert checks["API key"][0] is True
        assert checks["API key"][1] == "OpenAI + Anthropic"

    def test_all_missing(self, monkeypatch):
        def fake_run(cmd, *a, **k):
            raise FileNotFoundError()
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        monkeypatch.setattr(repairagent, "_check_python_packages", lambda: False)
        monkeypatch.setattr(repairagent, "_has_api_key", lambda k: False)
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)

        checks = repairagent.check_environment()
        assert checks["Java (JDK)"] == (False, "not found")
        assert checks["Perl"][0] is False
        assert checks["Python packages"] == (False, "some missing")
        assert checks["API key"] == (False, "not configured")

    def test_api_key_openai_only(self, monkeypatch):
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(1))
        monkeypatch.setattr(repairagent, "_check_python_packages", lambda: True)
        monkeypatch.setattr(repairagent, "_has_api_key",
                            lambda k: k == "OPENAI_API_KEY")
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        checks = repairagent.check_environment()
        assert checks["API key"] == (True, "OpenAI")


class TestDetectMissingSystemPackages:
    def test_all_present(self, monkeypatch):
        def fake_run(cmd, *a, **k):
            if cmd and cmd[0] == "java":
                return _completed(0, stderr="openjdk 11")
            return _completed(0, stdout="ok")
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        assert repairagent.detect_missing_system_packages() == []

    def test_all_missing(self, monkeypatch):
        def fake_run(cmd, *a, **k):
            raise FileNotFoundError()
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        missing = repairagent.detect_missing_system_packages()
        names = [pkg for pkg, _ in missing]
        assert names == list(repairagent.REQUIRED_SYSTEM_PACKAGES.keys())
        # each entry is (name, description)
        for pkg, desc in missing:
            assert isinstance(pkg, str) and isinstance(desc, str)

    def test_java_via_stderr_present(self, monkeypatch):
        # java returns nonzero but emits stderr -> counts as present (check_stderr)
        def fake_run(cmd, *a, **k):
            if cmd and cmd[0] == "java":
                return _completed(1, stderr='openjdk version "11"')
            raise FileNotFoundError()
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        missing = dict(repairagent.detect_missing_system_packages())
        assert "openjdk-11-jdk" not in missing
        assert "perl" in missing


# ===========================================================================
# 3. SETUP HELPERS (SCRIPT_DIR -> tmp_path)
# ===========================================================================

def _make_d4j_tree(root, projects):
    """Create a realistic defects4j/framework/projects/<P>/patches/<n>.src.patch tree.

    ``projects`` maps project name -> number of bugs. Each bug gets the real
    Defects4J file pair: "<n>.src.patch" (counted) and "<n>.test.patch" (decoy
    that must NOT be counted).
    """
    base = root / "defects4j" / "framework" / "projects"
    for name, count in projects.items():
        patches = base / name / "patches"
        patches.mkdir(parents=True)
        for i in range(1, count + 1):
            (patches / f"{i}.src.patch").write_text("patch")
            (patches / f"{i}.test.patch").write_text("not counted")
    return base


class TestGetAvailableProjects:
    def test_listing_excludes_lib(self, monkeypatch, tmp_path):
        base = _make_d4j_tree(tmp_path, {"Chart": 2, "Math": 1})
        (base / "lib").mkdir()
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent.get_available_projects() == ["Chart", "Math"]

    def test_no_projects_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent.get_available_projects() == []


class TestGetBugCount:
    def test_counts_only_src_patch_files(self, monkeypatch, tmp_path):
        # 3 bugs -> 3 ".src.patch" files (+ 3 ".test.patch" decoys that must not count)
        _make_d4j_tree(tmp_path, {"Chart": 3})
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent.get_bug_count("Chart") == 3

    def test_missing_project(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent.get_bug_count("Nope") == 0


class TestPrepareAiSettings:
    def test_writes_yaml(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        repairagent.prepare_ai_settings("Chart", "7")
        content = (tmp_path / "ai_settings.yaml").read_text()
        assert "Chart" in content
        assert '"7"' in content
        assert repairagent.AGENT_VERSION in content


class TestIncrementExperiment:
    def test_first_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        path = repairagent.increment_experiment()
        assert path.endswith("experiment_1")
        exp_dir = tmp_path / "experimental_setups" / "experiment_1"
        for sub in ["logs", "responses", "external_fixes", "saved_contexts",
                    "mutations_history", "plausible_patches"]:
            assert (exp_dir / sub).is_dir()
        list_txt = (tmp_path / "experimental_setups" / "experiments_list.txt")
        assert list_txt.read_text().strip() == "experiment_1"

    def test_increment_existing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        list_txt = tmp_path / "experimental_setups" / "experiments_list.txt"
        list_txt.parent.mkdir(parents=True)
        list_txt.write_text("experiment_1\nexperiment_2\n")
        path = repairagent.increment_experiment()
        assert path.endswith("experiment_3")
        assert "experiment_3" in list_txt.read_text()


class TestSetupDefects4jEnv:
    def test_path_and_perl5lib(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.delenv("PERL5LIB", raising=False)
        repairagent.setup_defects4j_env()
        d4j_bin = str(tmp_path / "defects4j" / "framework" / "bin")
        assert repairagent.os.environ["PATH"].startswith(d4j_bin)
        assert "perl5" in repairagent.os.environ["PERL5LIB"]
        assert repairagent.os.environ["LC_COLLATE"] == "C"

    def test_path_not_doubled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        d4j_bin = str(tmp_path / "defects4j" / "framework" / "bin")
        monkeypatch.setenv("PATH", d4j_bin + ":/usr/bin")
        repairagent.setup_defects4j_env()
        # should not be prepended a second time
        assert repairagent.os.environ["PATH"].count(d4j_bin) == 1


# ===========================================================================
# 4. CLI (CliRunner)
# ===========================================================================

class TestRunCommand:
    def test_bugs_option_parsed(self, monkeypatch):
        rec = {}

        def recorder(bugs, model, hyperparams, max_cycles, temperature=0.0):
            rec.update(bugs=bugs, model=model, hyperparams=hyperparams,
                       max_cycles=max_cycles, temperature=temperature)

        monkeypatch.setattr(repairagent, "execute_run", recorder)
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, [
            "run", "--bugs", "Chart 1,Math 5",
            "--model", "gpt-4o-mini", "--max-cycles", "5",
        ])
        assert result.exit_code == 0, result.output
        assert rec["bugs"] == [("Chart", "1"), ("Math", "5")]
        assert rec["model"] == "gpt-4o-mini"
        assert rec["max_cycles"] == 5

    def test_docker_flag_routes(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(repairagent, "run_in_docker",
                            lambda *a, **k: rec.update(called=True, args=a))
        # ensure execute_run is NOT called
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda *a, **k: rec.update(execute_called=True))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, [
            "run", "--bugs", "Chart 1", "--docker",
        ])
        assert result.exit_code == 0, result.output
        assert rec.get("called") is True
        assert "execute_called" not in rec
        assert rec["args"][0] == [("Chart", "1")]

    def test_bugs_file_path(self, monkeypatch, tmp_path):
        rec = {}
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda bugs, *a, **k: rec.update(bugs=bugs))
        f = tmp_path / "bugs.txt"
        f.write_text("Lang 2\nTime 3\n")
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["run", "--bugs-file", str(f)])
        assert result.exit_code == 0, result.output
        assert rec["bugs"] == [("Lang", "2"), ("Time", "3")]

    def test_no_bugs_falls_back_to_select_bugs(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(repairagent, "select_bugs",
                            lambda: [("Mockito", "9")])
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda bugs, *a, **k: rec.update(bugs=bugs))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["run"])
        assert result.exit_code == 0, result.output
        assert rec["bugs"] == [("Mockito", "9")]

    def test_empty_bugs_exits_1(self, monkeypatch):
        # --bugs with only invalid entries -> empty list -> sys.exit(1)
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda *a, **k: pytest.fail("should not run"))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["run", "--bugs", "garbage"])
        assert result.exit_code == 1

    def test_hyperparams_default(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda bugs, model, hyperparams, *a, **k:
                            rec.update(hp=hyperparams))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["run", "--bugs", "Chart 1"])
        assert result.exit_code == 0, result.output
        assert rec["hp"].endswith("hyperparams.json")


class TestSetupCommand:
    def test_all_ok(self, monkeypatch):
        ok_checks = {
            "Python 3.10+": (True, "3.11"),
            "API key": (True, "OpenAI"),
        }
        monkeypatch.setattr(repairagent, "check_environment", lambda: ok_checks)
        # Confirm.ask only reached for reconfigure -> say no
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: False)
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["setup"])
        assert result.exit_code == 0, result.output
        assert "All checks passed" in result.output

    def test_missing_deps_install(self, monkeypatch):
        # first call: something missing; after install: all ok
        states = iter([
            {"Python packages": (False, "some missing"), "API key": (True, "OpenAI")},
            {"Python packages": (True, "installed"), "API key": (True, "OpenAI")},
            {"Python packages": (True, "installed"), "API key": (True, "OpenAI")},
        ])
        monkeypatch.setattr(repairagent, "check_environment", lambda: next(states))
        rec = {}
        monkeypatch.setattr(repairagent, "install_all_dependencies",
                            lambda: rec.update(installed=True) or True)
        # Confirm.ask: install? yes; reconfigure? no
        answers = iter([True, False])
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: next(answers))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["setup"])
        assert result.exit_code == 0, result.output
        assert rec.get("installed") is True

    def test_docker_builds_image(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(repairagent, "check_environment",
                            lambda: {"API key": (True, "OpenAI")})
        monkeypatch.setattr(repairagent, "build_docker_image",
                            lambda: rec.update(built=True))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["setup", "--docker"])
        assert result.exit_code == 0, result.output
        assert rec.get("built") is True

    def test_no_api_key_configures(self, monkeypatch):
        states = iter([
            {"API key": (False, "not configured")},
            {"API key": (True, "OpenAI")},
        ])
        monkeypatch.setattr(repairagent, "check_environment", lambda: next(states))
        rec = {}
        monkeypatch.setattr(repairagent, "setup_api_keys",
                            lambda: rec.update(configured=True))
        # Confirm.ask: configure API keys now? -> yes
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, ["setup"])
        assert result.exit_code == 0, result.output
        assert rec.get("configured") is True


class TestBareCli:
    def test_invokes_interactive_run(self, monkeypatch):
        rec = {}
        monkeypatch.setattr(repairagent, "interactive_run",
                            lambda: rec.update(called=True))
        runner = CliRunner()
        result = runner.invoke(repairagent.cli, [])
        assert result.exit_code == 0, result.output
        assert rec.get("called") is True


# ===========================================================================
# Focused interactive_run happy path
# ===========================================================================

class TestInteractiveRun:
    def test_happy_path_local(self, monkeypatch):
        rec = {}
        # environment all OK including API key
        monkeypatch.setattr(repairagent, "check_environment",
                            lambda: {"API key": (True, "OpenAI")})
        # no docker -> behave as not-in-container but docker missing
        monkeypatch.setattr(repairagent, "is_in_container", lambda: False)
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        monkeypatch.setattr(repairagent, "select_model",
                            lambda: ("gpt-4o-mini", 0.0))
        monkeypatch.setattr(repairagent, "select_bugs",
                            lambda: [("Chart", "1")])
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 40)
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda bugs, model, hyperparams, max_cycles, temperature=0.0:
                            rec.update(bugs=bugs, model=model, max_cycles=max_cycles))

        repairagent.interactive_run()

        assert rec["bugs"] == [("Chart", "1")]
        assert rec["model"] == "gpt-4o-mini"
        assert rec["max_cycles"] == 40

    def test_abort_on_no_confirm(self, monkeypatch):
        monkeypatch.setattr(repairagent, "check_environment",
                            lambda: {"API key": (True, "OpenAI")})
        monkeypatch.setattr(repairagent, "is_in_container", lambda: True)
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        monkeypatch.setattr(repairagent, "select_model",
                            lambda: ("gpt-4o-mini", 0.0))
        monkeypatch.setattr(repairagent, "select_bugs",
                            lambda: [("Chart", "1")])
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 40)
        # Confirm "Start?" -> no
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: False)
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda *a, **k: pytest.fail("should not run"))
        # should return cleanly without running
        repairagent.interactive_run()

    def test_installs_missing_then_docker(self, monkeypatch):
        rec = {}
        # first check: missing python packages; after install: ok
        states = iter([
            {"Python packages": (False, "missing"), "API key": (False, "no")},
            {"Python packages": (True, "ok"), "API key": (True, "OpenAI")},
        ])
        monkeypatch.setattr(repairagent, "check_environment", lambda: next(states))
        monkeypatch.setattr(repairagent, "install_all_dependencies",
                            lambda: rec.update(installed=True))
        # API key still false on the FIRST checks snapshot used post-install?
        # interactive_run re-reads checks; second snapshot has API key True so no setup.
        monkeypatch.setattr(repairagent, "setup_api_keys",
                            lambda: rec.update(api=True))
        # in-container False + docker present -> mode prompt; pick "2" (docker)
        monkeypatch.setattr(repairagent, "is_in_container", lambda: False)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/docker")
        monkeypatch.setattr(repairagent, "select_model",
                            lambda: ("claude-x", 0.5))
        monkeypatch.setattr(repairagent, "select_bugs", lambda: [("Lang", "3")])
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 10)
        # Prompt.ask used for the mode choice -> "2"
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: "2")
        # Confirm: install missing? yes ; Start? yes
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        monkeypatch.setattr(repairagent, "run_in_docker",
                            lambda *a, **k: rec.update(docker=True, args=a))
        monkeypatch.setattr(repairagent, "execute_run",
                            lambda *a, **k: pytest.fail("should use docker"))

        repairagent.interactive_run()

        assert rec.get("installed") is True
        assert rec.get("docker") is True
        assert rec["args"][0] == [("Lang", "3")]


# ===========================================================================
# select_model
# ===========================================================================

class TestSelectModel:
    def test_openai_pick(self, monkeypatch):
        monkeypatch.setattr(repairagent, "_has_api_key",
                            lambda k: k == "OPENAI_API_KEY")
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 2)
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: "0.3")
        model, temp = repairagent.select_model()
        assert model == repairagent.OPENAI_MODELS[1][0]
        assert temp == 0.3

    def test_custom_model(self, monkeypatch):
        monkeypatch.setattr(repairagent, "_has_api_key", lambda k: True)
        n_models = len(repairagent.OPENAI_MODELS) + len(repairagent.ANTHROPIC_MODELS)
        custom_idx = n_models + 1
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: custom_idx)
        prompts = iter(["my-custom-model", "0.0"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        model, temp = repairagent.select_model()
        assert model == "my-custom-model"
        assert temp == 0.0

    def test_invalid_temperature_defaults(self, monkeypatch):
        monkeypatch.setattr(repairagent, "_has_api_key",
                            lambda k: k == "ANTHROPIC_API_KEY")
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 1)
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: "not-a-number")
        model, temp = repairagent.select_model()
        assert model == repairagent.ANTHROPIC_MODELS[0][0]
        assert temp == 0.0

    def test_index_clamped(self, monkeypatch):
        monkeypatch.setattr(repairagent, "_has_api_key",
                            lambda k: k == "OPENAI_API_KEY")
        # huge index -> clamped to custom (last) -> prompts for model name
        n = len(repairagent.OPENAI_MODELS) + 1
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 999)
        prompts = iter(["clamped-model", "1.0"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        model, temp = repairagent.select_model()
        assert model == "clamped-model"

    def test_no_keys_exits(self, monkeypatch):
        monkeypatch.setattr(repairagent, "_has_api_key", lambda k: False)
        with pytest.raises(SystemExit):
            repairagent.select_model()


# ===========================================================================
# select_bugs
# ===========================================================================

class TestSelectBugs:
    def test_manual(self, monkeypatch):
        # first Prompt.ask is the method choice ("Method"), second is the bugs str
        prompts = iter(["1", "Chart 1, Math 5"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        bugs = repairagent.select_bugs()
        assert bugs == [("Chart", "1"), ("Math", "5")]

    def test_project_range(self, monkeypatch, tmp_path):
        _make_d4j_tree(tmp_path, {"Chart": 5})
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        # Prompt: method="2", then range="2-4"
        prompts = iter(["2", "2-4"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 1)
        bugs = repairagent.select_bugs()
        assert bugs == [("Chart", "2"), ("Chart", "3"), ("Chart", "4")]

    def test_project_all(self, monkeypatch, tmp_path):
        _make_d4j_tree(tmp_path, {"Chart": 3})
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        prompts = iter(["2", "all"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 1)
        bugs = repairagent.select_bugs()
        assert bugs == [("Chart", "1"), ("Chart", "2"), ("Chart", "3")]

    def test_project_comma_list(self, monkeypatch, tmp_path):
        _make_d4j_tree(tmp_path, {"Chart": 5})
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        prompts = iter(["2", "1,3,5"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        monkeypatch.setattr(repairagent.IntPrompt, "ask", lambda *a, **k: 1)
        bugs = repairagent.select_bugs()
        assert bugs == [("Chart", "1"), ("Chart", "3"), ("Chart", "5")]

    def test_project_no_projects_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: "2")
        with pytest.raises(SystemExit):
            repairagent.select_bugs()

    def test_file(self, monkeypatch, tmp_path):
        f = tmp_path / "b.txt"
        f.write_text("Lang 9\n")
        prompts = iter(["3", str(f)])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        bugs = repairagent.select_bugs()
        assert bugs == [("Lang", "9")]

    def test_empty_exits(self, monkeypatch):
        prompts = iter(["1", "garbage-no-pair"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        with pytest.raises(SystemExit):
            repairagent.select_bugs()


# ===========================================================================
# execute_run
# ===========================================================================

class TestExecuteRun:
    def test_no_defects4j_exits(self, monkeypatch):
        monkeypatch.setattr(repairagent, "setup_defects4j_env", lambda: None)
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        with pytest.raises(SystemExit):
            repairagent.execute_run([("Chart", "1")], "m", "hp", 5)

    def test_success_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "setup_defects4j_env", lambda: None)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/defects4j")
        monkeypatch.setattr(repairagent, "generate_commands_descriptions",
                            lambda: None)
        monkeypatch.setattr(repairagent, "increment_experiment",
                            lambda: str(tmp_path / "experiment_1"))
        ran = []
        monkeypatch.setattr(repairagent, "run_single_bug",
                            lambda *a, **k: ran.append(a))
        repairagent.execute_run([("Chart", "1"), ("Math", "2")], "gpt", "hp", 7, 0.0)
        assert len(ran) == 2

    def test_agent_systemexit_zero_is_ok(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "setup_defects4j_env", lambda: None)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/defects4j")
        monkeypatch.setattr(repairagent, "generate_commands_descriptions",
                            lambda: None)
        monkeypatch.setattr(repairagent, "increment_experiment",
                            lambda: str(tmp_path / "exp"))

        def fake_run_single(*a, **k):
            raise SystemExit(0)
        monkeypatch.setattr(repairagent, "run_single_bug", fake_run_single)
        # should NOT raise — SystemExit(0) treated as normal completion
        repairagent.execute_run([("Chart", "1")], "gpt", "hp", 5)

    def test_agent_systemexit_one_reraises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "setup_defects4j_env", lambda: None)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/defects4j")
        monkeypatch.setattr(repairagent, "generate_commands_descriptions",
                            lambda: None)
        monkeypatch.setattr(repairagent, "increment_experiment",
                            lambda: str(tmp_path / "exp"))

        def fake_run_single(*a, **k):
            raise SystemExit(1)
        monkeypatch.setattr(repairagent, "run_single_bug", fake_run_single)
        with pytest.raises(SystemExit):
            repairagent.execute_run([("Chart", "1")], "gpt", "hp", 5)

    def test_exception_recorded_as_failed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "setup_defects4j_env", lambda: None)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/defects4j")
        monkeypatch.setattr(repairagent, "generate_commands_descriptions",
                            lambda: None)
        monkeypatch.setattr(repairagent, "increment_experiment",
                            lambda: str(tmp_path / "exp"))

        def fake_run_single(*a, **k):
            raise RuntimeError("boom")
        monkeypatch.setattr(repairagent, "run_single_bug", fake_run_single)
        # exception is caught and recorded; no raise
        repairagent.execute_run([("Chart", "1")], "gpt", "hp", 5)


# ===========================================================================
# run_in_docker
# ===========================================================================

class TestRunInDocker:
    def test_no_docker_exits(self, monkeypatch):
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        with pytest.raises(SystemExit):
            repairagent.run_in_docker([("Chart", "1")], "m", "hp", 5)

    def test_image_present_runs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/docker")
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            if "images" in cmd:
                return _completed(0, stdout="abc123")  # image exists
            return _completed(0)
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-2")
        repairagent.run_in_docker([("Chart", "1")], "gpt", "hp", 5, 0.1)
        # final docker run command issued
        run_cmd = calls[-1]
        assert run_cmd[0] == "docker" and "run" in run_cmd
        assert "Chart 1" in run_cmd
        # temp bugs file cleaned up
        assert not (tmp_path / ".tmp_bugs_list").exists()

    def test_image_missing_builds(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/docker")
        rec = {}
        monkeypatch.setattr(repairagent, "build_docker_image",
                            lambda: rec.update(built=True))
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-2")

        def fake_run(cmd, *a, **k):
            if "images" in cmd:
                return _completed(0, stdout="")  # image missing
            return _completed(0)
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        repairagent.run_in_docker([("Chart", "1")], "gpt", "hp", 5)
        assert rec.get("built") is True

    def test_image_missing_decline_exits(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/docker")
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: False)
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda cmd, *a, **k: _completed(0, stdout=""))
        with pytest.raises(SystemExit):
            repairagent.run_in_docker([("Chart", "1")], "gpt", "hp", 5)

    def test_keys_loaded_from_env_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/docker")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        (tmp_path / ".env").write_text(
            "# comment\nOPENAI_API_KEY=sk-fromfile\nANTHROPIC_API_KEY=GLOBAL-API-KEY-PLACEHOLDER\n"
        )
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            if "images" in cmd:
                return _completed(0, stdout="img")
            return _completed(0)
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        repairagent.run_in_docker([("Chart", "1")], "gpt", "hp", 5)
        run_cmd = calls[-1]
        assert "OPENAI_API_KEY=sk-fromfile" in run_cmd
        # placeholder not propagated
        assert "ANTHROPIC_API_KEY=GLOBAL-API-KEY-PLACEHOLDER" not in run_cmd


# ===========================================================================
# Misc small helpers
# ===========================================================================

class TestMiscHelpers:
    def test_display_environment_runs(self):
        # smoke: just ensure it renders without error
        repairagent.display_environment({
            "Python 3.10+": (True, "3.11"),
            "Java (JDK)": (False, "not found"),
        })

    def test_is_defects4j_initialized(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        assert repairagent.is_defects4j_initialized() is False
        d4j = tmp_path / "defects4j" / "framework" / "bin"
        d4j.mkdir(parents=True)
        (d4j / "defects4j").write_text("#!/bin/sh\n")
        assert repairagent.is_defects4j_initialized() is True

    def test_build_docker_image(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        calls = []
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda cmd, *a, **k: calls.append(cmd) or _completed(0))
        repairagent.build_docker_image()
        assert calls and calls[0][0] == "docker"

    def test_generate_commands_descriptions(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        calls = []
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda cmd, *a, **k: calls.append(cmd) or _completed(0))
        repairagent.generate_commands_descriptions()
        assert calls and "construct_commands_descriptions.py" in calls[0][-1]

    def test_checkout_bug(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        captured = {}
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda cmd, *a, **k: captured.update(cmd=cmd) or _completed(0))
        repairagent.checkout_bug("Chart", "3")
        assert "defects4j checkout" in captured["cmd"]
        assert "Chart" in captured["cmd"]

    def test_checkout_bug_failure_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)

        def boom(cmd, *a, **k):
            raise subprocess.CalledProcessError(1, cmd)
        monkeypatch.setattr(repairagent.subprocess, "run", boom)
        with pytest.raises(subprocess.CalledProcessError):
            repairagent.checkout_bug("Chart", "3")

    def test_install_python_requirements_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        (tmp_path / "requirements-core.txt").write_text("click\n")
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(0))
        assert repairagent.install_python_requirements() is True

    def test_install_python_requirements_failure(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        # no core file -> uses requirements.txt
        (tmp_path / "requirements.txt").write_text("click\n")
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(1))
        assert repairagent.install_python_requirements() is False

    def test_install_system_packages_no_apt(self, monkeypatch):
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        assert repairagent.install_system_packages(["perl"]) is False

    def test_install_system_packages_declined(self, monkeypatch):
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: "/usr/bin/apt-get")
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: False)
        assert repairagent.install_system_packages(["perl"]) is False

    def test_install_system_packages_success(self, monkeypatch):
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: "/usr/bin/apt-get")
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(0))
        assert repairagent.install_system_packages(["perl"]) is True

    def test_install_system_packages_update_fails(self, monkeypatch):
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: "/usr/bin/apt-get")
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda *a, **k: _completed(1))
        assert repairagent.install_system_packages(["perl"]) is False


# ===========================================================================
# install_all_dependencies orchestrator
# ===========================================================================

class TestInstallAllDependencies:
    def test_all_present(self, monkeypatch):
        monkeypatch.setattr(repairagent, "detect_missing_system_packages",
                            lambda: [])
        monkeypatch.setattr(repairagent, "_check_python_packages", lambda: True)
        monkeypatch.setattr(repairagent, "is_defects4j_initialized", lambda: True)
        assert repairagent.install_all_dependencies() is True

    def test_missing_all_user_installs(self, monkeypatch):
        monkeypatch.setattr(repairagent, "detect_missing_system_packages",
                            lambda: [("perl", "Perl")])
        monkeypatch.setattr(repairagent, "install_system_packages",
                            lambda pkgs: True)
        monkeypatch.setattr(repairagent, "_check_python_packages", lambda: False)
        monkeypatch.setattr(repairagent, "install_python_requirements",
                            lambda: True)
        monkeypatch.setattr(repairagent, "is_defects4j_initialized", lambda: False)
        monkeypatch.setattr(repairagent, "install_defects4j", lambda: True)
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: True)
        assert repairagent.install_all_dependencies() is True

    def test_missing_user_declines(self, monkeypatch):
        monkeypatch.setattr(repairagent, "detect_missing_system_packages",
                            lambda: [("perl", "Perl")])
        monkeypatch.setattr(repairagent, "install_system_packages",
                            lambda pkgs: False)
        monkeypatch.setattr(repairagent, "_check_python_packages", lambda: False)
        monkeypatch.setattr(repairagent, "is_defects4j_initialized", lambda: False)
        monkeypatch.setattr(repairagent.Confirm, "ask", lambda *a, **k: False)
        assert repairagent.install_all_dependencies() is False


# ===========================================================================
# setup_api_keys
# ===========================================================================

class TestSetupApiKeys:
    def test_openai_choice(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        # stub the set_api_key module functions
        import set_api_key
        monkeypatch.setattr(set_api_key, "set_env_var",
                            lambda *a, **k: None, raising=False)
        monkeypatch.setattr(set_api_key, "replace_placeholder",
                            lambda *a, **k: None, raising=False)
        # Prompt: provider "1", then the key
        prompts = iter(["1", "sk-openai-key"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = repairagent.setup_api_keys()
        assert provider == "openai"
        assert repairagent.os.environ["OPENAI_API_KEY"] == "sk-openai-key"
        assert (tmp_path / "token.txt").read_text() == "sk-openai-key"

    def test_both_choice(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        import set_api_key
        monkeypatch.setattr(set_api_key, "set_env_var",
                            lambda *a, **k: None, raising=False)
        monkeypatch.setattr(set_api_key, "replace_placeholder",
                            lambda *a, **k: None, raising=False)
        prompts = iter(["3", "sk-o", "sk-a"])
        monkeypatch.setattr(repairagent.Prompt, "ask", lambda *a, **k: next(prompts))
        provider = repairagent.setup_api_keys()
        assert provider == "both"


# ===========================================================================
# install_defects4j
# ===========================================================================

class TestInstallDefects4j:
    def test_full_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent, "DATA_DIR", tmp_path / "data")
        d4j_dir = tmp_path / "defects4j"

        # data dir with bug subdirs to exercise the copytree path
        for sub in ["buggy-lines", "buggy-methods"]:
            (tmp_path / "data" / sub).mkdir(parents=True)
            (tmp_path / "data" / sub / "x.txt").write_text("data")

        monkeypatch.setattr(repairagent.shutil, "which",
                            lambda x: "/usr/bin/cpanm")

        def fake_run(cmd, *a, **k):
            if cmd[:2] == ["git", "clone"]:
                # simulate clone creating the .git dir and init.sh
                (d4j_dir / ".git").mkdir(parents=True, exist_ok=True)
                (d4j_dir / "init.sh").write_text("#!/bin/sh\n")
            if cmd[:1] == ["bash"]:
                # simulate init producing the binary
                binp = d4j_dir / "framework" / "bin"
                binp.mkdir(parents=True, exist_ok=True)
                (binp / "defects4j").write_text("#!/bin/sh\n")
            return _completed(0)
        monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
        assert repairagent.install_defects4j() is True

    def test_clone_fails(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda cmd, *a, **k: _completed(1))
        assert repairagent.install_defects4j() is False

    def test_already_cloned_no_cpanm_init_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
        monkeypatch.setattr(repairagent, "DATA_DIR", tmp_path / "data")
        d4j_dir = tmp_path / "defects4j"
        (d4j_dir / ".git").mkdir(parents=True)  # already cloned
        # no cpanm available, no init.sh -> returns False at step 4
        monkeypatch.setattr(repairagent.shutil, "which", lambda x: None)
        monkeypatch.setattr(repairagent.subprocess, "run",
                            lambda cmd, *a, **k: _completed(0))
        assert repairagent.install_defects4j() is False


# ===========================================================================
# Bug-hunt regressions (Tier 2): dotenv parsing must match what the app loads.
# ===========================================================================
def test_has_api_key_detects_export_and_spaced_and_quoted_forms(monkeypatch, tmp_path):
    monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for content in ('export OPENAI_API_KEY=sk-real\n',
                    'OPENAI_API_KEY = sk-real\n',
                    'OPENAI_API_KEY="sk-real"\n'):
        (tmp_path / ".env").write_text(content)
        assert repairagent._has_api_key("OPENAI_API_KEY") is True, content


def test_run_in_docker_forwards_export_form_key(monkeypatch, tmp_path):
    monkeypatch.setattr(repairagent, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(repairagent.shutil, "which", lambda x: "/usr/bin/docker")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text('export OPENAI_API_KEY=sk-fromfile\n')
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        if "images" in cmd:
            return _completed(0, stdout="img")
        return _completed(0)
    monkeypatch.setattr(repairagent.subprocess, "run", fake_run)
    repairagent.run_in_docker([("Chart", "1")], "gpt", "hp", 5)
    assert "OPENAI_API_KEY=sk-fromfile" in calls[-1]
