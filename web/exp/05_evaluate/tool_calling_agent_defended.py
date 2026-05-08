#!/usr/bin/env python3
"""Web tool-calling agent with local prompt-injection defense baselines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
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
from utils.defenses import ALLOWED_DEFENSE_MODES, DEFENSE_MODE_DEFAULT
from utils.defenses.goal_reinforcement import (
    apply_step_wise_defense,
    apply_system_prompt_defense,
    build_goal_reinforcement_messages,
    merge_goal_reinforcement_into_input,
)
from utils.defenses.segmentation_removal import sanitize_observation_text


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
            )
        elif "OPENAI_API_KEY" in os.environ:
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)
        else:
            raise ValueError("Missing OpenAI API key")
        self.client = client
        self.model = model
        self.temperature = temperature
        self.seed = seed
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
            kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": self.tools_definitions,
            }
            if self.temperature is not None:
                kwargs["temperature"] = float(self.temperature)
            if self.seed is not None:
                kwargs["seed"] = int(self.seed)
            completion = self.client.chat.completions.create(**kwargs)
            print(
                f"Received model response. Used {completion.usage.prompt_tokens} prompt tokens and {completion.usage.completion_tokens} completion tokens"
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
            kwargs: dict[str, object] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            if self.seed is not None:
                kwargs["seed"] = int(self.seed)
            completion = self.client.chat.completions.create(**kwargs)
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
