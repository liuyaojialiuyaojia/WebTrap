import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "supplement_step_by_step_all_trials.py"
SPEC = importlib.util.spec_from_file_location(
    "supplement_step_by_step_all_trials_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
supplement = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supplement
SPEC.loader.exec_module(supplement)


def _assistant_tool_call(tool_call_id: str, name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        ],
    }


def _tool_result(tool_call_id: str, observation: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": observation,
    }


def _write_jsonl(path: Path, snapshots: list[list[dict]]) -> None:
    path.write_text(
        "\n".join(json.dumps(snapshot) for snapshot in snapshots) + "\n",
        encoding="utf-8",
    )


def test_parse_trace_filename_extracts_trial_number(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace_1001_r03.jsonl"
    trace_path.write_text("", encoding="utf-8")

    trace = supplement._parse_trace_filename(trace_path)

    assert trace.task_index == 1001
    assert trace.trial_number == 3
    assert trace.trial_name == "r03"


def test_validation_payload_detects_label_mismatch(tmp_path: Path) -> None:
    log_dir = tmp_path / "agent_logs_post_injection"
    log_dir.mkdir()

    canonical_df = pd.DataFrame(
        [
            {
                "task_index": 1000,
                "trial_name": "r01",
                "source_kind": "canonical_existing",
                "source_file": str(log_dir / "trace_1000_r01.jsonl"),
                "original_source_file": "canonical_logs/trace_1000.jsonl",
                "step_index": 0,
                "legitimate_objective": "Find the answer.",
                "action_description": "step 0",
                "injection_in_context_window": False,
                "classification_result": "not_compromised",
                "classification_result_reason": "ok",
            },
            {
                "task_index": 1000,
                "trial_name": "r01",
                "source_kind": "canonical_existing",
                "source_file": str(log_dir / "trace_1000_r01.jsonl"),
                "original_source_file": "canonical_logs/trace_1000.jsonl",
                "step_index": 1,
                "legitimate_objective": "Find the answer.",
                "action_description": "step 1",
                "injection_in_context_window": False,
                "classification_result": "compromised",
                "classification_result_reason": "bad",
            },
        ]
    )
    rerun_df = canonical_df.copy()
    rerun_df["source_kind"] = "supplemental_rerun"
    rerun_df.loc[1, "classification_result"] = "not_compromised"

    payload = supplement._validation_payload(
        canonical_df=canonical_df,
        rerun_df=rerun_df,
        log_dir=log_dir,
        validate_r01=True,
    )

    assert payload["status"] == "failed"
    assert payload["details"][0]["expected_labels"] == [
        "not_compromised",
        "compromised",
    ]
    assert payload["details"][0]["actual_labels"] == [
        "not_compromised",
        "not_compromised",
    ]


def test_build_trial_results_includes_trace_level_stop_stats(tmp_path: Path) -> None:
    log_dir = tmp_path / "agent_logs_post_injection"
    log_dir.mkdir()
    trace_path = log_dir / "trace_1000_r02.jsonl"

    system = {"role": "system", "content": "sys"}
    user = {"role": "user", "content": "Find the answer."}
    line0 = [
        system,
        user,
        _assistant_tool_call("call_goto", "goto", '{"url": "http://example.com"}'),
        _tool_result("call_goto", "OBSERVATION home"),
    ]
    line1 = [
        *line0,
        _assistant_tool_call("call_click", "click", '{"element_id": "42"}'),
        _tool_result("call_click", "OBSERVATION detail"),
    ]
    line2 = [
        *line1,
        _assistant_tool_call("call_stop", "stop", '{"answer": "done"}'),
        {"role": "stop", "answer": "done"},
    ]
    _write_jsonl(trace_path, [line0, line1, line2])

    df = pd.DataFrame(
        [
            {
                "task_index": 1000,
                "trial_name": "r02",
                "source_kind": "supplemental_rerun",
                "source_file": str(trace_path),
                "original_source_file": str(trace_path),
                "step_index": 0,
                "legitimate_objective": "Find the answer.",
                "action_description": "OBSERVATION home -> click",
                "injection_in_context_window": False,
                "classification_result": "compromised",
                "classification_result_reason": "bad",
            },
            {
                "task_index": 1000,
                "trial_name": "r02",
                "source_kind": "supplemental_rerun",
                "source_file": str(trace_path),
                "original_source_file": str(trace_path),
                "step_index": 1,
                "legitimate_objective": "Find the answer.",
                "action_description": "OBSERVATION detail -> stop",
                "injection_in_context_window": False,
                "classification_result": "not_compromised",
                "classification_result_reason": "ok",
            },
        ]
    )

    trial_results = supplement._build_trial_results(df)

    assert trial_results == [
        {
            "task_index": 1000,
            "trial_name": "r02",
            "source_kind": "supplemental_rerun",
            "trace_path": str(trace_path),
            "judged_action_count": 2,
            "total_tool_action_count": 3,
            "final_tool_name": "stop",
            "final_action_is_stop": True,
            "final_classification": "compromised",
            "first_compromised_step_index": 0,
            "first_compromised_step_number": 1,
        }
    ]
