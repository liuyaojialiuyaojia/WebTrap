#!/usr/bin/env python3
"""Web tool-calling agent with local prompt-injection defense baselines."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click
from openai import AzureOpenAI, OpenAI

REPO_ROOT = Path(__file__).resolve().parents[3]
VISUALWEBARENA_ROOT = REPO_ROOT / "wasp" / "visualwebarena"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VISUALWEBARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISUALWEBARENA_ROOT))
os.environ.setdefault("DATASET", "visualwebarena")
from experiment_defenses import (
    ALLOWED_DEFENSE_MODES,
    DEFENSE_MODE_DEFAULT,
    apply_step_wise_defense,
    apply_system_prompt_defense,
    build_goal_reinforcement_messages,
    merge_goal_reinforcement_into_input,
    sanitize_observation_text,
)

NO_CACHE_EXTRA_BODY = {
    "cache": {
        "no-cache": True,
        "no-store": True,
    }
}
RETRYABLE_OPENAI_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_OPENAI_RETRY_DELAYS_SECONDS = (5.0, 15.0, 60.0)
_INITIAL_OPENAI_JITTER_DONE = False


def _optional_positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer when set")
    return value


class OpenAIResponsePayloadError(RuntimeError):
    """Error returned inside a chat completion payload instead of as an exception."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _completion_usage_tokens(completion: object) -> tuple[object, object]:
    """Return prompt/completion token counts, tolerating proxy responses without usage."""

    usage = getattr(completion, "usage", None)
    return (
        getattr(usage, "prompt_tokens", "unknown"),
        getattr(usage, "completion_tokens", "unknown"),
    )


def _payload_value(payload: object, key: str) -> object:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _status_code_from_error_payload(payload: object, message: str) -> int | None:
    for key in ("status", "status_code", "code"):
        value = _payload_value(payload, key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for status_code in RETRYABLE_OPENAI_STATUS_CODES:
        if str(status_code) in message:
            return status_code
    return None


def _raise_for_completion_error_payload(completion: object) -> None:
    """Raise when a proxy encodes an API error in a nominal completion object."""

    payload = _payload_value(completion, "error")
    if payload:
        message = str(_payload_value(payload, "message") or payload)
        raise OpenAIResponsePayloadError(
            message,
            status_code=_status_code_from_error_payload(payload, message),
        )
    if _payload_value(completion, "choices") is None:
        raise OpenAIResponsePayloadError(
            "OpenAI response payload did not include choices",
            status_code=503,
        )


def _build_completion_kwargs(
    *,
    model: str,
    messages: list[dict],
    temperature: float | None,
    seed: int | None,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
) -> dict[str, object]:
    """Build provider-portable Chat Completions arguments.

    OpenRouter/LiteLLM may ignore unsupported optional parameters such as
    ``seed`` for a particular provider. The cache directives keep repeated
    ablation trials independent, while the tool settings force the WebArena
    agent's one-action-at-a-time contract.
    """

    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "extra_body": {
            "cache": dict(NO_CACHE_EXTRA_BODY["cache"]),
        },
    }
    reasoning_effort = os.environ.get("WEBTRAP_REASONING_EFFORT")
    if reasoning_effort:
        kwargs["extra_body"]["reasoning_effort"] = reasoning_effort
    if tools is not None:
        kwargs.update(
            {
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            }
        )
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if seed is not None:
        kwargs["seed"] = int(seed)
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    return kwargs


def _openai_retry_delays() -> tuple[float, ...]:
    raw = os.environ.get("WEBTRAP_OPENAI_RETRY_DELAYS")
    if not raw:
        return DEFAULT_OPENAI_RETRY_DELAYS_SECONDS
    delays: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        delays.append(max(0.0, float(item)))
    return tuple(delays)


def _openai_request_timeout() -> float | None:
    raw = os.environ.get("WEBTRAP_OPENAI_TIMEOUT")
    if not raw:
        return None
    return max(1.0, float(raw))


def _openai_retry_jitter_seconds() -> float:
    raw = os.environ.get("WEBTRAP_OPENAI_RETRY_JITTER_SECONDS")
    if not raw:
        return 0.0
    return max(0.0, float(raw))


def _openai_initial_jitter_seconds() -> float:
    raw = os.environ.get("WEBTRAP_OPENAI_INITIAL_JITTER_SECONDS")
    if not raw:
        return 0.0
    return max(0.0, float(raw))


@contextmanager
def _openai_api_lock():
    lock_path = os.environ.get("WEBTRAP_OPENAI_LOCK_FILE")
    if not lock_path:
        yield
        return

    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _is_retryable_openai_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in RETRYABLE_OPENAI_STATUS_CODES:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        if body.get("retryable") is True:
            return True
        body_status = body.get("status")
        if body_status in RETRYABLE_OPENAI_STATUS_CODES:
            return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status in RETRYABLE_OPENAI_STATUS_CODES:
        return True

    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "bad gateway",
            "cloudflare",
            "rate limit",
            "retry shortly",
            "saturated",
            "serviceunavailable",
            "temporarily unavailable",
            "timeout",
        )
    )


def _create_chat_completion_with_retries(client, **kwargs):
    global _INITIAL_OPENAI_JITTER_DONE
    delays = _openai_retry_delays()
    for attempt_index in range(len(delays) + 1):
        initial_jitter = _openai_initial_jitter_seconds()
        if attempt_index == 0 and initial_jitter and not _INITIAL_OPENAI_JITTER_DONE:
            _INITIAL_OPENAI_JITTER_DONE = True
            delay = random.uniform(0.0, initial_jitter)
            print(
                f"Staggering initial OpenAI API request by {delay:g}s",
                flush=True,
            )
            time.sleep(delay)
        try:
            with _openai_api_lock():
                completion = client.chat.completions.create(**kwargs)
            _raise_for_completion_error_payload(completion)
            return completion
        except Exception as exc:
            if attempt_index >= len(delays) or not _is_retryable_openai_error(exc):
                raise
            delay = delays[attempt_index]
            jitter = _openai_retry_jitter_seconds()
            if jitter:
                delay += random.uniform(0.0, jitter)
            print(
                "Retryable OpenAI API error "
                f"on attempt {attempt_index + 1}/{len(delays) + 1}: {exc}. "
                f"Retrying in {delay:g}s",
                flush=True,
            )
            time.sleep(delay)


def _load_visualwebarena_runtime():
    from browser_env import ScriptBrowserEnv
    from browser_env.auto_login import get_site_comb_from_filepath
    from tool_calling_utils import (
        SYSTEM_PROMPT,
        TOOL_NAME_TO_CREATE_ACTION_FUNCTION,
        WEB_TOOLS_DEFINITION,
    )

    return {
        "ScriptBrowserEnv": ScriptBrowserEnv,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "TOOL_NAME_TO_CREATE_ACTION_FUNCTION": TOOL_NAME_TO_CREATE_ACTION_FUNCTION,
        "WEB_TOOLS_DEFINITION": WEB_TOOLS_DEFINITION,
        "get_site_comb_from_filepath": get_site_comb_from_filepath,
    }


def _normalize_tool_call_args(tool_name: str, args: dict[str, object]) -> dict[str, object]:
    normalized_args = dict(args)
    if tool_name in {"click", "type", "hover"}:
        if "id" in normalized_args and "element_id" not in normalized_args:
            normalized_args["element_id"] = normalized_args.pop("id")
        if "element_id" in normalized_args and not isinstance(
            normalized_args["element_id"], str
        ):
            normalized_args["element_id"] = str(normalized_args["element_id"])
    return normalized_args


class GPTWebAgent:
    def __init__(
        self,
        model: str,
        filepath_to_trace_log: str,
        temperature: float | None = None,
        seed: int | None = None,
        defense_mode: str = DEFENSE_MODE_DEFAULT,
    ):
        if defense_mode not in ALLOWED_DEFENSE_MODES:
            raise ValueError(
                f"Invalid defense_mode='{defense_mode}'. Expected one of: {', '.join(ALLOWED_DEFENSE_MODES)}"
            )
        if "AZURE_API_ENDPOINT" in os.environ and "AZURE_API_KEY" in os.environ:
            api_version = (
                "2024-10-21"
                if "AZURE_API_VERSION" not in os.environ
                else os.environ["AZURE_API_VERSION"]
            )
            client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_API_ENDPOINT"],
                api_key=os.environ["AZURE_API_KEY"],
                api_version=api_version,
                timeout=_openai_request_timeout(),
            )
        elif "OPENAI_API_KEY" in os.environ:
            client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
                "OPENAI_API_BASE"
            )
            if base_url:
                client_kwargs["base_url"] = base_url
            request_timeout = _openai_request_timeout()
            if request_timeout is not None:
                client_kwargs["timeout"] = request_timeout
            client = OpenAI(**client_kwargs)
        else:
            raise ValueError("Missing OpenAI API key")
        self.client = client
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = _optional_positive_int_env("WEBTRAP_AGENT_MAX_TOKENS")
        self.defense_mode = defense_mode
        self.defense_cache: dict[tuple[object, ...], object] = {}
        self.current_user_objective = ""
        self.runtime = _load_visualwebarena_runtime()
        self.tools_definitions = self.runtime["WEB_TOOLS_DEFINITION"]
        self.system_prompt = self.runtime["SYSTEM_PROMPT"]
        self.tool_name_to_action = self.runtime["TOOL_NAME_TO_CREATE_ACTION_FUNCTION"]
        script_browser_env = self.runtime["ScriptBrowserEnv"]

        self.browser_env = script_browser_env(
            headless=True,
            slow_mo=0,
            observation_type="accessibility_tree",
            current_viewport_only=False,
            viewport_size={
                "width": 1280,
                "height": 2048,
            },
            save_trace_enabled=False,
            sleep_after_execution=0.0,
            captioning_fn=None,
        )

        self.filepath_to_trace_log = filepath_to_trace_log

    def __enter__(self):
        self.browser_env.reset()
        self.trace_log_file = open(self.filepath_to_trace_log, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.browser_env.close()
        self.trace_log_file.close()

    def _tool_message(self, tool_call_id: str, content: str) -> dict:
        content = merge_goal_reinforcement_into_input(
            content,
            self.defense_mode,
            trusted_goal=self.current_user_objective,
            medium="screen",
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": apply_step_wise_defense(
                content, self.defense_mode, medium="screen"
            ),
        }

    def _call_model(self, messages: list[dict]):
        try:
            kwargs = _build_completion_kwargs(
                model=self.model,
                messages=messages,
                tools=self.tools_definitions,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
            )
            completion = _create_chat_completion_with_retries(self.client, **kwargs)
            prompt_tokens, completion_tokens = _completion_usage_tokens(completion)
            print(
                f"Received model response. Used {prompt_tokens} prompt tokens and {completion_tokens} completion tokens"
            )
            return _parse_response_to_json(completion.choices[0].message)
        except Exception as exc:
            print(f"Error occurred while requesting OpenAI API: {exc}")
            return {
                "role": "assistant",
                "content": f"Error occurred while requesting OpenAI API: {exc}",
                "tool_calls": [],
            }

    def _ask_defense_model(
        self, messages: list[dict[str, str]], max_tokens: int
    ) -> str | None:
        try:
            kwargs = _build_completion_kwargs(
                model=self.model,
                messages=messages,
                temperature=0.0,
                seed=self.seed,
                max_tokens=max_tokens,
            )
            completion = _create_chat_completion_with_retries(self.client, **kwargs)
            response = completion.choices[0].message.content
            return str(response or "").strip()
        except Exception as exc:
            print(f"[defense] auxiliary model call failed: {exc}")
            return None

    def _sanitize_observation_text(self, observation_text: str) -> str:
        return sanitize_observation_text(
            observation_text,
            defense_mode=self.defense_mode,
            user_goal=self.current_user_objective,
            scene="web",
            ask_model=self._ask_defense_model,
            cache=self.defense_cache,
        )

    def _execute_requested_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        if len(tool_calls) > 1:
            return [
                self._tool_message(
                    tool_call["id"],
                    "ERROR: Multiple tool calls provided. You MUST respond with ONLY one tool call!",
                )
                for tool_call in tool_calls
            ]

        if len(tool_calls) < 1:
            return []

        tool_call = tool_calls[0]
        tool_name = tool_call["function"]["name"]
        if tool_name not in self.tool_name_to_action:
            return [
                self._tool_message(
                    tool_call["id"],
                    f"ERROR: {tool_name} is not a valid function! You must pick one of {','.join(self.tool_name_to_action.keys())}",
                )
            ]

        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            return [self._tool_message(tool_call["id"], f"ERROR: {exc}")]
        if not isinstance(args, dict):
            return [
                self._tool_message(
                    tool_call["id"], "ERROR: Tool arguments must be a JSON object."
                )
            ]
        args = _normalize_tool_call_args(tool_name, args)

        if tool_name == "stop":
            return [{"role": "stop", "answer": args["answer"]}]

        create_action_function = self.tool_name_to_action[tool_name]
        try:
            action = create_action_function(**args)
        except Exception as exc:
            return [self._tool_message(tool_call["id"], f"ERROR: {exc}")]

        browser_execution_result = self.browser_env.step(action)
        observation_text = browser_execution_result[0]["text"]
        observation_text = self._sanitize_observation_text(observation_text)

        formatted_tool_call_result = f"""OBSERVATION:
{observation_text}
URL: {self.browser_env.page.url}
"""

        return [self._tool_message(tool_call["id"], formatted_tool_call_result)]

    def _log_messages(self, messages: list[dict]) -> None:
        self.trace_log_file.write(json.dumps(messages) + "\n")

    def loop(
        self,
        start_url: str,
        user_objective: str,
        max_actions: int,
        max_observations_to_keep: int,
    ) -> None:
        self.current_user_objective = user_objective
        messages: list[dict] = []
        system_message = {
            "role": "system",
            "content": apply_system_prompt_defense(
                self.system_prompt, self.defense_mode, medium="screen"
            ),
        }
        messages.append(system_message)

        user_intent_message = {
            "role": "user",
            "content": apply_step_wise_defense(
                f"Start on {start_url} {user_objective}",
                self.defense_mode,
                medium="screen",
            ),
        }
        messages.append(user_intent_message)

        for action_number in range(max_actions):
            model_response_message = self._call_model(messages)

            print(
                f"Model [{self.model}] response {action_number} {model_response_message['content']}"
            )

            result_of_execution = self._execute_requested_tool_calls(
                model_response_message["tool_calls"]
            )

            messages.append(model_response_message)

            if len(result_of_execution) < 1:
                print("Agent did not call any tools; exiting.")
                break

            messages.extend(result_of_execution)
            if result_of_execution[0]["role"] != "stop":
                messages.extend(
                    build_goal_reinforcement_messages(
                        self.defense_mode,
                        trusted_goal=user_objective,
                        medium="screen",
                    )
                )

            self._log_messages(messages)

            if result_of_execution[0]["role"] == "stop":
                print(
                    f"Agent finished with stop action and answer {result_of_execution[0]['answer']}"
                )
                break
            _maybe_filter_tool_call_results(messages, max_observations_to_keep)


def _maybe_filter_tool_call_results(messages: list[dict], max_observations_to_keep: int):
    counter = 0
    indices_of_removal = []
    tool_call_ids_to_remove = set()
    for original_index, message in reversed(list(enumerate(messages))):
        if message["role"] == "tool":
            counter += 1
            if counter > max_observations_to_keep:
                indices_of_removal.append(original_index)
                tool_call_ids_to_remove.add(message["tool_call_id"])
        elif message["role"] == "assistant":
            if message["tool_calls"] and any(
                tool_call["id"] in tool_call_ids_to_remove
                for tool_call in message["tool_calls"]
            ):
                indices_of_removal.append(original_index)

    for index_to_remove in indices_of_removal:
        messages.pop(index_to_remove)


def _parse_response_to_json(response_message):
    parsed_message = {
        "role": response_message.role,
        "content": response_message.content,
        "tool_calls": [],
    }

    if response_message.tool_calls:
        parsed_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in response_message.tool_calls
        ]

    return parsed_message


@click.command()
@click.option(
    "--webarena_config_path", type=str, help="path to the json config describing the task"
)
@click.option("--model", type=str, default="gpt-4o", help="The model backing the agent")
@click.option(
    "--temperature",
    type=float,
    default=None,
    help="Optional sampling temperature (default: API default).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Optional sampling seed (default: unset).",
)
@click.option(
    "--defense-mode",
    type=click.Choice(list(ALLOWED_DEFENSE_MODES), case_sensitive=True),
    default=DEFENSE_MODE_DEFAULT,
    show_default=True,
    help="Defense mode for prompt injection.",
)
@click.option(
    "--trace-log-filepath",
    type=str,
    default="/tmp/gpt_text_loop_agent_logs.jsonl",
    help="Where to store the trajectories",
)
@click.option("--max_actions", type=int, default=20, help="The cap on actions by the agent")
@click.option(
    "--max_observations_to_keep",
    type=int,
    default=3,
    help="The maximum number of past tool call results to keep",
)
def main(
    webarena_config_path,
    model,
    temperature,
    seed,
    defense_mode,
    trace_log_filepath,
    max_actions,
    max_observations_to_keep,
):
    with open(webarena_config_path, encoding="utf-8") as handle:
        config = json.load(handle)
        start_url = config["start_url"]
        user_objective = config["intent"]
        if config["storage_state"]:
            try:
                runtime = _load_visualwebarena_runtime()
                cookie_file_name = os.path.basename(config["storage_state"])
                comb = runtime["get_site_comb_from_filepath"](cookie_file_name)
                temp_dir = tempfile.mkdtemp()
                auto_login_path = VISUALWEBARENA_ROOT / "browser_env" / "auto_login.py"
                subprocess.run(
                    [
                        "python",
                        str(auto_login_path),
                        "--auth_folder",
                        temp_dir,
                        "--site_list",
                        *comb,
                    ],
                    check=False,
                )
                config["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                assert os.path.exists(config["storage_state"])
                config_file = f"{temp_dir}/{os.path.basename(webarena_config_path)}"
                with open(config_file, "w", encoding="utf-8") as rewritten:
                    json.dump(config, rewritten)
            except Exception as exc:
                print(f"Failed to automatically log in: {exc}")
                print("Ignore this failure since agent has credentials in the system_prompt")

    with GPTWebAgent(
        model,
        trace_log_filepath,
        temperature=temperature,
        seed=seed,
        defense_mode=defense_mode,
    ) as agent:
        agent.loop(
            start_url=start_url,
            user_objective=user_objective,
            max_actions=max_actions,
            max_observations_to_keep=max_observations_to_keep,
        )


if __name__ == "__main__":
    main()
