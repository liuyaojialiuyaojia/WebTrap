import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "supplement_reasoning_eval_all_trials.py"
SPEC = importlib.util.spec_from_file_location(
    "supplement_reasoning_eval_all_trials_under_test",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
supplement = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supplement
SPEC.loader.exec_module(supplement)


def _write_trace(path: Path, snapshots: list[list[dict]]) -> None:
    path.write_text(
        "\n".join(json.dumps(snapshot) for snapshot in snapshots) + "\n",
        encoding="utf-8",
    )


def test_rows_to_final_label_and_first_compromised_step() -> None:
    rows = [
        {"step_index": 0, "classification_result": "not_compromised"},
        {"step_index": 1, "classification_result": "compromised"},
    ]
    assert supplement._rows_to_final_label(rows) == "compromised"
    assert supplement._first_compromised_step(rows) == (1, 2)


def test_build_canonical_trial_map_matches_r01(tmp_path: Path) -> None:
    log_dir = tmp_path / "agent_logs_post_injection"
    canonical_dir = log_dir / "canonical_logs"
    canonical_dir.mkdir(parents=True)

    trace_r01 = log_dir / "trace_1000_r01.jsonl"
    trace_r02 = log_dir / "trace_1000_r02.jsonl"
    canonical = canonical_dir / "trace_1000.jsonl"

    snapshots = [
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Find info"},
            {
                "role": "assistant",
                "content": "Let's think step-by-step. Go to page.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "goto", "arguments": '{"url":"http://example.com"}'},
                    }
                ],
            },
        ]
    ]
    _write_trace(trace_r01, snapshots)
    _write_trace(canonical, snapshots)
    _write_trace(trace_r02, snapshots + snapshots)

    mapping = supplement._build_canonical_trial_map(log_dir)
    assert mapping == {1000: 1}


def test_augment_rows_rewrites_source_file(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace_1000_r01.jsonl"
    trace_path.write_text("", encoding="utf-8")
    trace = supplement.TraceRef(
        log_dir=tmp_path,
        trace_path=trace_path,
        task_index=1000,
        trial_number=1,
    )
    rows = [
        {
            "task_index": 1000,
            "source_file": "canonical_logs/trace_1000.jsonl",
            "step_index": 0,
            "legitimate_objective": "Find info",
            "action_description": "reasoning",
            "tool_name": "goto",
            "tool_call_id": "call_0",
            "classification_result": "not_compromised",
            "classification_result_reason": "ok",
        }
    ]
    augmented = supplement._augment_rows(rows, trace=trace, source_kind="reused_r01")
    assert augmented[0]["trial_name"] == "r01"
    assert augmented[0]["source_kind"] == "reused_r01"
    assert augmented[0]["original_source_file"] == "canonical_logs/trace_1000.jsonl"
    assert augmented[0]["source_file"].endswith("trace_1000_r01.jsonl")
