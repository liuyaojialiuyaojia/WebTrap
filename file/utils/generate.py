"""Utilities for calling OpenAI-compatible LLM APIs."""

from __future__ import annotations

import json
import time
import warnings
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset, concatenate_datasets
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

try:
    from utils.llm_client import chat as llm_chat

    _LLM_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    llm_chat = None  # type: ignore[assignment]
    _LLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class ChatCompletionMessageWithUsage(ChatCompletionMessage):
    usage: Optional[CompletionUsage] = None


def generate(
    messages: List[Dict[str, Any]],
    model: str,
    num_retries: int = 5,
    delay: int = 1,
    **kwargs,
) -> Union[ChatCompletionMessageWithUsage, None]:
    """Call the configured model API and return response message with usage metadata."""

    if num_retries <= 0:
        warnings.warn(
            f"Failed to get response from model {model}: num_retries must be > 0",
            stacklevel=2,
        )
        return None

    if llm_chat is None:  # pragma: no cover
        warnings.warn(
            "LLM client is unavailable. "
            f"Import error: {_LLM_IMPORT_ERROR or 'unknown'}",
            stacklevel=2,
        )
        return None

    wait_seconds = max(delay, 0)
    for attempt in range(1, num_retries + 1):
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        try:
            response = llm_chat(
                model=model,
                messages=messages,
                **kwargs,
            )
            if not response.choices:
                raise ValueError("No response choices from model API")

            original_response: ChatCompletionMessage = response.choices[0].message
            usage_info: Optional[CompletionUsage] = getattr(response, "usage", None)
            return ChatCompletionMessageWithUsage(
                **original_response.model_dump(exclude_unset=False),
                usage=usage_info,
            )
        except Exception as exc:
            if attempt >= num_retries:
                warnings.warn(
                    f"Failed to get response from model {model} after {num_retries} retries: {exc}",
                    stacklevel=2,
                )
                return None
            warnings.warn(
                f"LLM call failed (attempt {attempt}/{num_retries}) for model {model}: {exc}. Retrying...",
                stacklevel=2,
            )
            wait_seconds = max(wait_seconds, 1) * 2

    return None


def generate_to_dataset(
    dataset: Dataset,
    models: list[str],
    target_column: str = "prompt",
    **kwargs,
) -> Dataset:
    """Generate responses to a HuggingFace dataset of prompts."""

    generated_datasets = []
    for model_name in models:

        def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
            prompt_content = example[target_column]
            current_messages = [{"role": "user", "content": prompt_content}]
            api_response_message = generate(
                messages=current_messages, model=model_name, **kwargs
            )

            if api_response_message:
                response_text = (
                    api_response_message.content if api_response_message.content else ""
                )
                if api_response_message.tool_calls:
                    response_text += (
                        " (Tool calls: "
                        f"{json.dumps([tc.model_dump() for tc in api_response_message.tool_calls])})"
                    )
                return {"response": response_text}
            return {"response": "Error: No response from model"}

        generated_dataset = dataset.map(
            process_example,
        )
        generated_dataset = generated_dataset.add_column(
            "model", len(generated_dataset) * [model_name]
        )
        generated_datasets.append(generated_dataset)

    if not generated_datasets:
        return Dataset.from_dict({"response": [], "model": [], target_column: []})
    dataset = concatenate_datasets(generated_datasets)
    return dataset
