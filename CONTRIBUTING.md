# Contributing to RepairAgent

Thanks for your interest in improving RepairAgent! Bug reports, fixes, tests, and
ideas are all welcome.

## Dev setup (no Java, Defects4J, or API key needed)

The unit-test suite **stubs the heavy runtime stack**, so you can develop and run
the tests without a full repair environment:

```bash
cd repair_agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest tests -q
```

To run the **actual agent** on Defects4J bugs you do need Java 11, Defects4J, and an
API key — see the README "Quick Start". Check your environment with:

```bash
python3 repairagent.py doctor
```

## Before opening a pull request

- `pytest tests` passes (run from `repair_agent/`).
- Formatting is clean: `black --check tests` and `isort --check-only tests`
  (CI enforces this on `tests/`; the legacy `autogpt/` sources are not reformatted).
- **Add or update a test for any behavior change.** We follow red, then green: a
  test that fails before your change and passes after it.
- Keep each PR focused on one concern, with a short, imperative commit message
  (e.g. `fix: ...`, `feat: ...`, `test: ...`).

## Coverage

CI enforces a coverage floor on the core modules (see `repair_agent/.coveragerc`):

```bash
coverage run -m pytest tests && coverage report
```

## Reporting bugs

For a failed repair run, please include (the issue form will prompt you):
the exact command, the model, how you installed (Codespaces / Dev Container /
Docker / Local), your OS and Python version, the output of
`python3 repairagent.py doctor`, and relevant logs from
`experimental_setups/<experiment>/logs` or `.../responses`.

## Project layout

| Path | What it is |
|------|------------|
| `repair_agent/repairagent.py` | CLI entry point and interactive setup wizard |
| `repair_agent/autogpt/` | Agent framework, prompt construction, Defects4J commands |
| `repair_agent/tests/` | Pytest suite (`conftest.py` stubs `langchain` and sets up paths) |
| `repair_agent/experimental_setups/` | Per-run outputs and analysis scripts |
