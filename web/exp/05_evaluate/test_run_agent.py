import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "run_agent.py"
SPEC = importlib.util.spec_from_file_location("run_agent_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_agent = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_agent
SPEC.loader.exec_module(run_agent)


def test_build_command_reuses_current_python_interpreter(tmp_path: Path) -> None:
    command = run_agent._build_command(
        task_path=tmp_path / "1000.json",
        model="test-model",
        temperature=1.0,
        seed=42,
        defense_mode="default_attack",
        max_actions=20,
        max_observations=20,
        log_path=tmp_path / "trace.jsonl",
    )

    assert command[0] == sys.executable
