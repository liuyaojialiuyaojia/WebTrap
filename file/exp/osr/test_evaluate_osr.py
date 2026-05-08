import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "evaluate_osr.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_osr_under_test",
    MODULE_PATH,
)
evaluate_osr = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(evaluate_osr)


def test_evaluate_osr_prefers_lowest_sample_idx_for_one_time(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    logs_dir = run_dir / "logs"
    eval_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    (run_dir / "attack_cases.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"case_id": "case_0001"}),
                json.dumps({"case_id": "case_0002"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "user_goal_per_case.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"case_id": "case_0001", "user_goal_success": True}),
                json.dumps({"case_id": "case_0002", "user_goal_success": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "user_goal_per_sample.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "case_0001",
                        "sample_idx": 0,
                        "run_status": "ok",
                        "user_goal_success": False,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case_0002",
                        "sample_idx": 0,
                        "run_status": "ok",
                        "user_goal_success": False,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case_0001",
                        "sample_idx": 1,
                        "run_status": "ok",
                        "user_goal_success": True,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case_0002",
                        "sample_idx": 1,
                        "run_status": "ok",
                        "user_goal_success": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs_dir / "run_case_0001.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (logs_dir / "run_case_0002.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    metrics = evaluate_osr.evaluate_osr(run_dir)

    assert metrics["canonical_run"]["user_task_success_rate"] == 0.5
    assert metrics["one_time"]["selected_sample_idx"] == 0
    assert metrics["one_time"]["user_task_success_rate"] == 0.0
    assert metrics["best_of_n"]["user_task_success_rate"] == 0.5
    assert metrics["per_sample"]["1"]["user_task_success_rate"] == 0.5
