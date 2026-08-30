import json
from pathlib import Path

from Rebuttal.experiments.run_web_model_ablation import (
    _apply_user_intent_prefix,
    assess_compromise_checkpoint,
    build_agent_env,
    discover_gitlab_attacks,
    model_path_name,
    validate_agent_outputs,
)


def test_model_path_name_preserves_provider_and_model() -> None:
    assert (
        model_path_name("anthropic/claude-haiku-4.5")
        == "anthropic__claude-haiku-4.5"
    )


def test_discover_gitlab_attacks_excludes_other_suites(tmp_path: Path) -> None:
    source = tmp_path / "batch"
    attacker_root = tmp_path / "attacker"
    gitlab = source / "wasp_gitlab_add_member"
    reddit = source / "wasp_reddit_post"
    gitlab.joinpath("static_snapshot").mkdir(parents=True)
    reddit.joinpath("static_snapshot").mkdir(parents=True)
    gitlab.joinpath("static_snapshot/page_metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )
    reddit.joinpath("static_snapshot/page_metadata.json").write_text(
        "{}",
        encoding="utf-8",
    )
    attacker_root.joinpath(gitlab.name).mkdir(parents=True)

    attacks = discover_gitlab_attacks(source, attacker_root)

    assert [name for name, _metadata, _tasks in attacks] == [gitlab.name]


def test_validate_agent_outputs_rejects_silent_api_error(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    worker_logs = log_dir / "worker_logs"
    worker_logs.mkdir(parents=True)
    trace = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "click", "arguments": "{}"},
                }
            ],
        },
    ]
    log_dir.joinpath("trace_1000.jsonl").write_text(
        json.dumps(trace) + "\n",
        encoding="utf-8",
    )
    log_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": 1000,
                        "trials": [{"trial": 1, "status": "completed"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    worker_logs.joinpath("task_1000.log").write_text(
        "Error occurred while requesting OpenAI API: bad request",
        encoding="utf-8",
    )

    validation = validate_agent_outputs(
        log_dir=log_dir,
        task_ids=[1000],
        trials=1,
    )

    assert validation["passed"] is False
    assert any("model API error" in error for error in validation["errors"])


def test_build_agent_env_overrides_only_agent_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("GPT_AGENT_KEY", "secret-agent-key")
    args = type(
        "Args",
        (),
        {
            "agent_api_base_url": "https://llm.example/openai/v1",
            "agent_api_key_env": "GPT_AGENT_KEY",
            "agent_reasoning_effort": "none",
        },
    )()
    base_env = {
        "OPENAI_BASE_URL": "http://localhost:4000/v1",
        "OPENAI_API_BASE": "http://localhost:4000/v1",
        "OPENAI_API_KEY": "checker-key",
    }

    agent_env = build_agent_env(args, base_env)

    assert base_env["OPENAI_API_KEY"] == "checker-key"
    assert agent_env["OPENAI_API_KEY"] == "secret-agent-key"
    assert agent_env["OPENAI_BASE_URL"] == "https://llm.example/openai/v1"
    assert agent_env["OPENAI_API_BASE"] == "https://llm.example/openai/v1"
    assert agent_env["WEBTRAP_REASONING_EFFORT"] == "none"


def test_apply_user_intent_prefix_updates_numeric_tasks_only(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    task_dir.joinpath("1000.json").write_text(
        json.dumps({"intent": "original task", "task_id": 1000}),
        encoding="utf-8",
    )
    task_dir.joinpath("manifest.json").write_text(
        json.dumps({"tasks": [{"task_id": 1000}]}),
        encoding="utf-8",
    )

    _apply_user_intent_prefix(task_dir, "prefix: ")

    task = json.loads(task_dir.joinpath("1000.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        task_dir.joinpath("manifest.json").read_text(encoding="utf-8")
    )
    assert task["intent"] == "prefix: original task"
    assert manifest == {"tasks": [{"task_id": 1000}]}


def _write_checkpoint_sample(
    root: Path,
    *,
    name: str,
    attacker_passed: int,
    compromised_trials: int,
) -> None:
    sample = root / name
    sample.joinpath(
        "agent_logs_post_injection/reasoning_eval_all_trials"
    ).mkdir(parents=True)
    sample.joinpath("metrics_post_injection.json").write_text(
        json.dumps(
            {
                "attacker_utility": {
                    "total_scores": attacker_passed,
                    "cnt_tasks": 2,
                },
                "user_utility": {
                    "total_scores": 2,
                    "cnt_tasks": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    sample.joinpath("protocol.json").write_text(
        json.dumps({"task_ids": [1000, 1001], "trials": 3}),
        encoding="utf-8",
    )
    labels = (
        ["compromised"] * compromised_trials
        + ["not_compromised"] * (6 - compromised_trials)
    )
    sample.joinpath(
        "agent_logs_post_injection/reasoning_eval_all_trials/trial_results.json"
    ).write_text(
        json.dumps(
            {
                "counts": {
                    "compromised": compromised_trials,
                    "not_compromised": 6 - compromised_trials,
                },
                "trials": [
                    {"binary_final_label": label}
                    for label in labels
                ],
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_stops_only_when_both_attack_metrics_are_zero(
    tmp_path: Path,
) -> None:
    names = ["sample_a", "sample_b", "sample_c"]
    for name in names:
        _write_checkpoint_sample(
            tmp_path,
            name=name,
            attacker_passed=0,
            compromised_trials=0,
        )

    checkpoint = assess_compromise_checkpoint(
        run_root=tmp_path,
        attack_names=names,
    )

    assert checkpoint["intermediate_total_trials"] == 18
    assert checkpoint["completely_uncompromised"] is True
    assert checkpoint["decision"] == "stop_zero_compromise"


def test_checkpoint_continues_for_intermediate_compromise(
    tmp_path: Path,
) -> None:
    names = ["sample_a", "sample_b", "sample_c"]
    for index, name in enumerate(names):
        _write_checkpoint_sample(
            tmp_path,
            name=name,
            attacker_passed=0,
            compromised_trials=1 if index == 2 else 0,
        )

    checkpoint = assess_compromise_checkpoint(
        run_root=tmp_path,
        attack_names=names,
    )

    assert checkpoint["attacker_passed_tasks"] == 0
    assert checkpoint["intermediate_compromised_trials"] == 1
    assert checkpoint["completely_uncompromised"] is False
    assert checkpoint["decision"] == "continue_nonzero_compromise"
