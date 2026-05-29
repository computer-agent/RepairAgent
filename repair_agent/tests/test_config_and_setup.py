"""Tests for autogpt/config/config.py (Config + credentials) and the pure
``list_java_files`` setup helper in create_files_index.py.

Config is a pydantic-v1 model (via SystemSettings). The base SystemSettings
requires ``name`` and ``description``; everything else has defaults, so a minimal
Config(name=..., description=...) is enough to construct.

No network/LLM calls are made here: get_openai_credentials / get_azure_credentials
are pure dict-builders that only read attributes off the model.
"""
import os

import pytest

from autogpt.config.config import GPT_3_MODEL, GPT_4_MODEL, Config
from autogpt.llm.providers.openai import ALL_CHAT_MODELS
from create_files_index import list_java_files


def _minimal_config(**overrides):
    base = dict(name="test-config", description="config for tests")
    base.update(overrides)
    return Config(**base)


# --------------------------------------------------------------------------- #
# Non-azure get_openai_credentials                                            #
# --------------------------------------------------------------------------- #
def test_non_azure_credentials_only_basic_keys():
    cfg = _minimal_config(
        openai_api_key="sk-abc",
        openai_api_base="https://api.openai.com/v1",
        openai_organization="org-123",
    )
    creds = cfg.get_openai_credentials("gpt-4")

    # Exactly the three non-azure keys, nothing azure-specific.
    assert creds == {
        "api_key": "sk-abc",
        "api_base": "https://api.openai.com/v1",
        "organization": "org-123",
    }
    assert "api_type" not in creds
    assert "deployment_id" not in creds
    assert "engine" not in creds


def test_non_azure_credentials_defaults_are_none():
    cfg = _minimal_config()
    creds = cfg.get_openai_credentials("gpt-4")
    assert creds == {"api_key": None, "api_base": None, "organization": None}
    assert cfg.use_azure is False


# --------------------------------------------------------------------------- #
# Azure get_openai_credentials / get_azure_credentials                        #
# --------------------------------------------------------------------------- #
def _azure_config(dep_map, **overrides):
    base = dict(
        use_azure=True,
        fast_llm="gpt-3.5-turbo-0125",
        smart_llm="gpt-4-0314",
        openai_api_key="sk-azure",
        openai_api_type="azure",
        openai_api_base="https://example.openai.azure.com",
        openai_api_version="2023-03-15-preview",
        azure_model_to_deployment_id_map=dep_map,
    )
    base.update(overrides)
    return _minimal_config(**base)


def test_azure_credentials_merge_basic_and_azure_keys():
    cfg = _azure_config(
        {
            "fast_llm_deployment_id": "fast-dep",
            "smart_llm_deployment_id": "smart-dep",
            "embedding_model_deployment_id": "emb-dep",
        }
    )
    creds = cfg.get_openai_credentials("gpt-3.5-turbo-0125")
    # basic keys preserved
    assert creds["api_key"] == "sk-azure"
    assert creds["api_base"] == "https://example.openai.azure.com"
    assert creds["organization"] is None
    # azure keys added
    assert creds["api_type"] == "azure"
    assert creds["api_version"] == "2023-03-15-preview"
    assert creds["deployment_id"] == "fast-dep"


def test_azure_smart_llm_resolves_smart_deployment():
    cfg = _azure_config(
        {
            "fast_llm_deployment_id": "fast-dep",
            "smart_llm_deployment_id": "smart-dep",
        }
    )
    creds = cfg.get_azure_credentials("gpt-4-0314")
    assert creds["deployment_id"] == "smart-dep"
    assert "engine" not in creds


def test_azure_embedding_model_uses_engine_key_not_deployment_id():
    cfg = _azure_config(
        {
            "fast_llm_deployment_id": "fast-dep",
            "smart_llm_deployment_id": "smart-dep",
            "embedding_model_deployment_id": "emb-dep",
        }
    )
    creds = cfg.get_azure_credentials(cfg.embedding_model)
    # For the embedding model the code uses "engine" instead of "deployment_id".
    assert creds["engine"] == "emb-dep"
    assert "deployment_id" not in creds


def test_azure_unknown_model_gives_none_deployment_id():
    cfg = _azure_config({"fast_llm_deployment_id": "fast-dep"})
    creds = cfg.get_azure_credentials("some-unknown-model")
    assert creds["deployment_id"] is None


def test_azure_backwards_compat_fast_llm_model_deployment_id():
    # Only the legacy "*_model_deployment_id" key is present.
    cfg = _azure_config({"fast_llm_model_deployment_id": "legacy-fast"})
    creds = cfg.get_azure_credentials("gpt-3.5-turbo-0125")
    assert creds["deployment_id"] == "legacy-fast"


def test_azure_new_key_takes_precedence_over_legacy():
    cfg = _azure_config(
        {
            "fast_llm_deployment_id": "new-fast",
            "fast_llm_model_deployment_id": "legacy-fast",
        }
    )
    creds = cfg.get_azure_credentials("gpt-3.5-turbo-0125")
    assert creds["deployment_id"] == "new-fast"


def test_azure_gpt3only_alias_collision_renames_smart_llm_key():
    # When fast_llm == smart_llm and both start with GPT_3_MODEL, the smart_llm
    # key is renamed to f"not_{smart_llm}" so it cannot collide with fast_llm in
    # the lookup dict. The fast_llm lookup must still resolve normally.
    same = GPT_3_MODEL  # "gpt-3.5-turbo-0125"
    cfg = _azure_config(
        {
            "fast_llm_deployment_id": "fast-dep",
            "smart_llm_deployment_id": "smart-dep",
        },
        fast_llm=same,
        smart_llm=same,
    )
    creds = cfg.get_azure_credentials(same)
    # The real model string maps to the fast_llm deployment (smart key got renamed).
    assert creds["deployment_id"] == "fast-dep"


def test_azure_gpt4only_alias_collision_renames_fast_llm_key():
    # Mirror case: both == a GPT-4 model. fast_llm key gets renamed, so a lookup
    # of the shared model string resolves via smart_llm deployment.
    same = "gpt-4-0314"  # starts with GPT_4_MODEL
    assert same.startswith(GPT_4_MODEL)
    cfg = _azure_config(
        {
            "fast_llm_deployment_id": "fast-dep",
            "smart_llm_deployment_id": "smart-dep",
        },
        fast_llm=same,
        smart_llm=same,
    )
    creds = cfg.get_azure_credentials(same)
    assert creds["deployment_id"] == "smart-dep"


# --------------------------------------------------------------------------- #
# validators                                                                  #
# --------------------------------------------------------------------------- #
def test_openai_functions_validator_rejects_non_function_model():
    # A claude model has supports_functions=False, so enabling openai_functions
    # with it as smart_llm must fail validation.
    assert ALL_CHAT_MODELS["claude-sonnet"].supports_functions is False
    with pytest.raises(Exception) as exc:  # pydantic ValidationError
        _minimal_config(openai_functions=True, smart_llm="claude-sonnet")
    assert "does not support" in str(exc.value)


def test_openai_functions_validator_preserves_true_for_supported_model():
    # The validator now returns the value, so a supported model keeps the flag True.
    assert ALL_CHAT_MODELS["gpt-4"].supports_functions is True
    cfg = _minimal_config(openai_functions=True, smart_llm="gpt-4")
    assert cfg.openai_functions is True


def test_openai_functions_validator_skips_unknown_model():
    # smart_llm not in ALL_CHAT_MODELS -> the support assertion is skipped, but the
    # value is preserved (no longer collapses to None).
    assert "gpt-4-0314" not in ALL_CHAT_MODELS
    cfg = _minimal_config(openai_functions=True, smart_llm="gpt-4-0314")
    assert cfg.openai_functions is True


def test_openai_functions_default_false_passes_through():
    cfg = _minimal_config()
    # The validator is not "always", so it does not run on the unspecified default;
    # the field keeps its declared default of False.
    assert cfg.openai_functions is False


def test_plugins_field_rejects_non_plugin_objects():
    # The plugins field is typed list[AutoGPTPluginTemplate]; arbitrary objects
    # fail pydantic's arbitrary-type check before the custom validator runs.
    with pytest.raises(Exception) as exc:
        _minimal_config(plugins=[object()])
    assert "AutoGPTPluginTemplate" in str(exc.value)


def test_plugins_default_is_empty_list():
    cfg = _minimal_config()
    assert cfg.plugins == []


# --------------------------------------------------------------------------- #
# list_java_files (pure setup helper)                                         #
# --------------------------------------------------------------------------- #
def test_list_java_files_filters_extension_and_strips_subdir_prefix(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "A.java").write_text("")
    (tmp_path / "sub" / "B.java").write_text("")
    (tmp_path / "notes.txt").write_text("")

    result = list_java_files(str(tmp_path))

    # Only .java files are returned.
    assert all(p.endswith(".java") for p in result)
    assert len(result) == 2

    # All files are returned relative to main_dir (relpath), nested and top-level.
    assert os.path.join("sub", "B.java") in result
    assert "A.java" in result


def test_list_java_files_empty_when_no_java(tmp_path):
    (tmp_path / "x.txt").write_text("")
    (tmp_path / "y.py").write_text("")
    assert list_java_files(str(tmp_path)) == []
