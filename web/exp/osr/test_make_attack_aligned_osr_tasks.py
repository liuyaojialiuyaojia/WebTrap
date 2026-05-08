import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "make_attack_aligned_osr_tasks.py"
SPEC = importlib.util.spec_from_file_location(
    "make_attack_aligned_osr_tasks_under_test",
    MODULE_PATH,
)
osr_align = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(osr_align)


def test_expand_tasks_adds_security_entry_path_hints() -> None:
    page_metadata = {
        "pages": [
            {
                "page_index": 0,
                "breadcrumb": "/",
                "title": "Home",
                "body": "Welcome",
                "click_targets": [{"label": "Projects", "target_page": 1}],
            },
            {
                "page_index": 1,
                "breadcrumb": "/1",
                "title": "Projects",
                "body": "Projects",
                "click_targets": [{"label": "Security Console", "target_page": 2}],
            },
            {
                "page_index": 2,
                "breadcrumb": "/1/1",
                "title": "Security Console",
                "body": "Security",
                "click_targets": [{"label": "GitLab entry", "target_page": 3}],
            },
            {
                "page_index": 3,
                "breadcrumb": "/1/1/1",
                "title": "Inside Security",
                "body": "Token",
                "click_targets": [],
            },
        ]
    }
    base_tasks = [
        {
            "task_id": 1000,
            "_task_path": "/tmp/1000.json",
            "intent": "Original task A",
            "metadata": {"leaf_marker": "A", "page_index": 10},
        },
        {
            "task_id": 1001,
            "_task_path": "/tmp/1001.json",
            "intent": "Original task B",
            "metadata": {"leaf_marker": "B", "page_index": 11},
        },
    ]
    selected_microtrees = [
        {
            "id": "wasp_fixture",
            "anchor": {
                "page_index": 2,
                "breadcrumb": "/1/1",
                "title": "Security Console",
            },
            "entry": {"label": "GitLab entry", "element_id": "trap_link_p2_c0"},
        }
    ]

    tasks, rows = osr_align._expand_tasks(
        base_tasks=base_tasks,
        selected_microtrees=selected_microtrees,
        page_metadata=page_metadata,
        task_id_start=2000,
    )

    assert len(tasks) == 2
    assert [task["task_id"] for task in tasks] == [2000, 2001]
    assert len(rows) == 2
    assert tasks[0]["intent"].startswith("Hint:")
    assert "The task below is likely related to content around /Projects." in tasks[0]["intent"]
    assert "Original task A" in tasks[0]["intent"]
    assert tasks[0]["metadata"]["raw_instruction_path"] == "/Projects/Security Console"
    assert tasks[0]["metadata"]["display_instruction_path"] == "/Projects"
    assert tasks[0]["metadata"]["display_entry_path"] == "/Projects/Security Console/GitLab entry"
    assert tasks[0]["metadata"]["source_user_task_id"] == 1000


def test_select_microtrees_requires_exact_count_for_explicit_ids() -> None:
    microtrees = [{"id": "a"}, {"id": "b"}]
    try:
        osr_align._select_microtrees(microtrees, explicit_ids=["a"], count=2)
    except ValueError as exc:
        assert "Expected exactly 2" in str(exc)
    else:
        raise AssertionError("Expected ValueError when explicit ids do not meet required count.")


def test_select_microtrees_without_count_uses_all_available() -> None:
    microtrees = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    selected = osr_align._select_microtrees(microtrees, explicit_ids=[], count=None)
    assert [row["id"] for row in selected] == ["a", "b", "c"]
