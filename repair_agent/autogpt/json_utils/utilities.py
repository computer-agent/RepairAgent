"""Utilities for the json_fixes package."""
import ast
import json
import os.path
from json_repair import repair_json
from typing import Any, Literal

from jsonschema import Draft7Validator

from autogpt.config import Config
from autogpt.logs import logger

LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"


def _iter_balanced_objects(text: str):
    """Yield each top-level balanced ``{...}`` substring of ``text``.

    String-literal aware: braces (and backticks, newlines, ``` fences) that
    appear inside a JSON string value are ignored, so an opening ``` fence on
    the same line, multiple code blocks, and ``` inside a value do not derail
    extraction. Nested objects are part of the enclosing top-level object, not
    yielded separately.
    """
    i, n = 0, len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            in_str = False
            esc = False
            for j in range(i, n):
                c = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            yield text[i:j + 1]
                            i = j
                            break
            else:
                # No matching close brace; nothing more to find.
                return
        i += 1


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    if not response_content or not response_content.strip():
        return {}

    # Try each balanced {...} object found in the text (this transparently
    # handles fenced code blocks, surrounding prose, multiple blocks, and
    # backticks/newlines inside string values), then fall back to repairing the
    # whole response. The first candidate that parses to a non-empty dict wins.
    candidates = list(_iter_balanced_objects(response_content))
    candidates.append(response_content)

    for candidate in candidates:
        repaired = repair_json(candidate)
        if not repaired or not repaired.strip():
            continue
        # Try json.loads first (handles null/true/false correctly), then fall
        # back to ast.literal_eval for Python-dict-style responses.
        try:
            result = json.loads(repaired)
            if isinstance(result, dict) and result:
                return result
        except json.JSONDecodeError:
            pass
        try:
            result = ast.literal_eval(repaired)
            if isinstance(result, dict) and result:
                return result
        except BaseException:
            pass

    # Surface this as a warning, not debug: a model whose output cannot be parsed
    # into a command dict produces a no-op cycle, which is a common cause of the
    # agent looping without making progress (especially with non-OpenAI / Azure
    # deployments that do not reliably honor response_format=json_object).
    logger.warn(f"Could not parse model response as dict: {response_content[:200]}")
    return {}


def llm_response_schema(
    config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(filename, "r") as f:
        try:
            json_schema = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON schema: {e}")
    if config.openai_functions:
        del json_schema["properties"]["command"]
        json_schema["required"].remove("command")
    return json_schema


def validate_dict(
    object: object, config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> tuple[Literal[True], None] | tuple[Literal[False], list]:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object

    Returns:
        bool: Whether the json_object is valid or not
        list: Errors found in the json_object, or None if the object is valid
    """
    schema = llm_response_schema(config, schema_name)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(object), key=lambda e: e.path):
        for error in errors:
            logger.debug(f"JSON Validation Error: {error}")

        if config.debug_mode:
            logger.error(json.dumps(object, indent=4))
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
        return False, errors

    logger.debug("The JSON object is valid.")

    return True, None
