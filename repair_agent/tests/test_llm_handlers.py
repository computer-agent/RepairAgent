"""Tests for autogpt/llm/utils/__init__.py — newer-model / Azure handling.

Covers the two pure helpers (`_looks_like_reasoning_model`,
`_apply_reasoning_model_kwargs`) and the retry logic in
`create_chat_completion` (monkeypatching the recorded underlying call
`autogpt.llm.utils.iopenai.create_chat_completion`).

No real network/LLM calls are made; the underlying provider call is replaced
with a recording fake.
"""

from types import SimpleNamespace

import openai
import pytest

import autogpt.llm.utils as llm_utils
from autogpt.llm.utils import (
    _apply_reasoning_model_kwargs,
    _looks_like_reasoning_model,
)


# --------------------------------------------------------------------------- #
# _looks_like_reasoning_model                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5.2",
        "gpt-5-mini",
        "o1-preview",
        "o3-mini",
        "o4-mini",
        "my-azure-gpt-5-deployment",  # azure deployment name containing 'gpt-5'
    ],
)
def test_looks_like_reasoning_model_true(model):
    assert _looks_like_reasoning_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "claude-sonnet-4-6",
        "",
        None,
    ],
)
def test_looks_like_reasoning_model_false(model):
    assert _looks_like_reasoning_model(model) is False


def test_looks_like_reasoning_model_is_case_insensitive():
    assert _looks_like_reasoning_model("GPT-5-Mini") is True
    assert _looks_like_reasoning_model("O1-Preview") is True


# --------------------------------------------------------------------------- #
# _apply_reasoning_model_kwargs                                               #
# --------------------------------------------------------------------------- #
def test_apply_reasoning_model_kwargs_full():
    kwargs = {
        "max_tokens": 123,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "model": "gpt-5",
    }
    result = _apply_reasoning_model_kwargs(kwargs)

    # mutates in place, returns None
    assert result is None
    assert kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert "response_format" not in kwargs
    # unrelated keys are preserved
    assert kwargs["model"] == "gpt-5"


def test_apply_reasoning_model_kwargs_no_max_tokens():
    kwargs = {"temperature": 0.7, "response_format": {"type": "json_object"}}
    _apply_reasoning_model_kwargs(kwargs)

    # nothing added when max_tokens absent
    assert "max_completion_tokens" not in kwargs
    assert "max_tokens" not in kwargs
    # temperature / response_format still dropped
    assert "temperature" not in kwargs
    assert "response_format" not in kwargs


def test_apply_reasoning_model_kwargs_idempotent_on_clean_dict():
    kwargs = {"model": "gpt-5", "messages": []}
    _apply_reasoning_model_kwargs(kwargs)
    assert kwargs == {"model": "gpt-5", "messages": []}


# --------------------------------------------------------------------------- #
# create_chat_completion retry logic                                          #
# --------------------------------------------------------------------------- #
class _RecordingCompletion:
    """Records each call's kwargs; returns a canned response on success."""

    def __init__(self, errors=None):
        # errors: list of exceptions (or None) per successive call
        self.errors = list(errors or [])
        self.calls = []

    def __call__(self, *, messages, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        if idx < len(self.errors) and self.errors[idx] is not None:
            raise self.errors[idx]
        return SimpleNamespace(
            choices=[SimpleNamespace(message={"content": "ok", "function_call": None})]
        )


def _make_prompt(model):
    return SimpleNamespace(
        model=SimpleNamespace(name=model),
        token_length=10,
        raw=lambda: [],
        dump=lambda: "",
    )


def _make_config():
    return SimpleNamespace(
        temperature=0.0,
        plugins=[],
        get_openai_credentials=lambda model: {},
    )


def _invalid_request(message):
    # openai 0.27.x InvalidRequestError(message, param, ...)
    return openai.error.InvalidRequestError(message, param=None)


@pytest.fixture(autouse=True)
def _clear_reasoning_cache():
    """The module caches discovered reasoning models in a global set;
    clear it so tests don't leak into one another."""
    before = set(llm_utils._REASONING_MODELS)
    llm_utils._REASONING_MODELS.clear()
    yield
    llm_utils._REASONING_MODELS.clear()
    llm_utils._REASONING_MODELS.update(before)


# When an InvalidRequestError signals a parameter incompatibility (the wording
# differs between public OpenAI and Azure OpenAI), the call should retry ONCE
# with the reasoning-model kwargs (max_completion_tokens, no temperature/
# response_format) and succeed.
def test_openai_style_unsupported_max_tokens_triggers_one_retry(monkeypatch):
    err = _invalid_request(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    fake = _RecordingCompletion(errors=[err, None])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    resp = llm_utils.create_chat_completion(
        prompt=_make_prompt("gpt-4o"),
        config=_make_config(),
        model="gpt-4o",
        max_tokens=500,
    )

    assert resp.content == "ok"
    assert len(fake.calls) == 2

    first, second = fake.calls
    # first call used the standard params
    assert "max_tokens" in first
    assert "temperature" in first
    assert "response_format" in first
    # retry used the reasoning-model params
    assert second["max_completion_tokens"] == 500
    assert "max_tokens" not in second
    assert "temperature" not in second
    assert "response_format" not in second


def test_azure_style_temperature_not_supported_triggers_retry(monkeypatch):
    err = _invalid_request("temperature is not supported with this model.")
    fake = _RecordingCompletion(errors=[err, None])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    resp = llm_utils.create_chat_completion(
        prompt=_make_prompt("gpt-4o"),
        config=_make_config(),
        model="gpt-4o",
        max_tokens=321,
    )

    assert resp.content == "ok"
    assert len(fake.calls) == 2
    second = fake.calls[1]
    assert second["max_completion_tokens"] == 321
    assert "temperature" not in second
    assert "response_format" not in second


def test_retry_caches_model_to_skip_future_round_trips(monkeypatch):
    """After a parameter-incompatibility retry, the model is remembered in
    ``_REASONING_MODELS`` so subsequent calls go straight to the compatible
    kwargs instead of paying the failed round-trip again."""
    err = _invalid_request(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    fake = _RecordingCompletion(errors=[err, None])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    assert "gpt-4o" not in llm_utils._REASONING_MODELS
    resp = llm_utils.create_chat_completion(
        prompt=_make_prompt("gpt-4o"),
        config=_make_config(),
        model="gpt-4o",
        max_tokens=500,
    )

    # the retry succeeded (standard attempt + one retry) ...
    assert resp.content == "ok"
    assert len(fake.calls) == 2
    # ... and the model is now cached so the next call skips the round-trip
    assert "gpt-4o" in llm_utils._REASONING_MODELS


def test_unrelated_invalid_request_is_reraised_without_retry(monkeypatch):
    err = _invalid_request("Incorrect API key provided: invalid api key")
    fake = _RecordingCompletion(errors=[err])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    with pytest.raises(openai.error.InvalidRequestError):
        llm_utils.create_chat_completion(
            prompt=_make_prompt("gpt-4o"),
            config=_make_config(),
            model="gpt-4o",
            max_tokens=500,
        )

    # no retry attempted
    assert len(fake.calls) == 1


def test_reasoning_model_uses_compatible_kwargs_on_first_call(monkeypatch):
    fake = _RecordingCompletion(errors=[None])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    resp = llm_utils.create_chat_completion(
        prompt=_make_prompt("gpt-5.2"),
        config=_make_config(),
        model="gpt-5.2",
        max_tokens=200,
    )

    assert resp.content == "ok"
    # only one call, no exception/retry needed
    assert len(fake.calls) == 1
    first = fake.calls[0]
    assert "temperature" not in first
    assert "response_format" not in first
    assert "max_tokens" not in first
    assert first["max_completion_tokens"] == 200
    # model got cached as a reasoning model
    assert "gpt-5.2" in llm_utils._REASONING_MODELS


def test_cached_reasoning_model_skips_retry_round_trip(monkeypatch):
    # Pre-seed the cache: a model that does NOT look like a reasoning model by
    # name but was previously discovered to be one.
    llm_utils._REASONING_MODELS.add("custom-deploy")
    fake = _RecordingCompletion(errors=[None])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    llm_utils.create_chat_completion(
        prompt=_make_prompt("custom-deploy"),
        config=_make_config(),
        model="custom-deploy",
        max_tokens=77,
    )

    assert len(fake.calls) == 1
    first = fake.calls[0]
    assert first["max_completion_tokens"] == 77
    assert "temperature" not in first


def test_already_converted_kwargs_not_double_retried(monkeypatch):
    # A reasoning model that STILL raises an unsupported-param error on the
    # (already converted) retry kwargs must NOT loop: the guard checks for
    # 'max_completion_tokens' already present, so it re-raises.
    err = _invalid_request(
        "Unsupported parameter: 'max_completion_tokens' is not supported."
    )
    fake = _RecordingCompletion(errors=[err])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    with pytest.raises(openai.error.InvalidRequestError):
        llm_utils.create_chat_completion(
            prompt=_make_prompt("gpt-5"),
            config=_make_config(),
            model="gpt-5",
            max_tokens=200,
        )

    # gpt-5 converts on first call -> max_completion_tokens present -> no retry
    assert len(fake.calls) == 1


# ===========================================================================
# Bug-hunt regressions in the reasoning-model retry logic.
# ===========================================================================
def test_cache_not_poisoned_when_retry_fails(monkeypatch):
    # A genuine param error triggers the retry, but the retry fails for an
    # UNRELATED reason. The model must NOT be cached as a reasoning model
    # (we never confirmed the param-rewrite was correct).
    err1 = _invalid_request(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
    )
    err2 = _invalid_request("The server had an error while processing your request.")
    fake = _RecordingCompletion(errors=[err1, err2])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    with pytest.raises(openai.error.InvalidRequestError):
        llm_utils.create_chat_completion(
            prompt=_make_prompt("gpt-4o"),
            config=_make_config(),
            model="gpt-4o",
            max_tokens=500,
        )
    assert "gpt-4o" not in llm_utils._REASONING_MODELS


def test_unrelated_error_mentioning_temperature_is_not_retried(monkeypatch):
    # An error that merely contains the words "temperature" and "unsupported"
    # but is NOT a parameter rejection must be re-raised, not retried (which
    # would mask the real error).
    err = _invalid_request(
        "Your prompt about reactor temperature was flagged: unsupported content."
    )
    fake = _RecordingCompletion(errors=[err, None])
    monkeypatch.setattr(llm_utils.iopenai, "create_chat_completion", fake)

    with pytest.raises(openai.error.InvalidRequestError):
        llm_utils.create_chat_completion(
            prompt=_make_prompt("gpt-4o"),
            config=_make_config(),
            model="gpt-4o",
            max_tokens=500,
        )
    assert len(fake.calls) == 1  # no retry
