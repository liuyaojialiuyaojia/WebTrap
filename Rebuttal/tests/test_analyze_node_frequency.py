import json
from pathlib import Path

from Rebuttal.experiments.analyze_node_frequency import (
    TraceRecord,
    _normalize_file_path,
    file_candidate_sets,
    parse_browser_trace,
    parse_file_trace,
    summarize_coverage,
)


def test_parse_browser_trace_deduplicates_pages(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                "URL: http://127.0.0.1:8124/index.html",
                "URL: http://127.0.0.1:8124/index.html#p7",
                "URL: http://127.0.0.1:8124/index.html#p7",
                "URL: http://127.0.0.1:8124/index.html#p9",
            ]
        ),
        encoding="utf-8",
    )
    assert parse_browser_trace(trace) == frozenset({0, 7, 9})


def test_parse_file_trace_counts_directories_and_read_files(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tool_call": {"name": "cd_dir"},
                        "tool_observation": {
                            "success": True,
                            "current_path": "/alpha",
                        },
                    }
                ),
                json.dumps(
                    {
                        "tool_call": {"name": "read_file"},
                        "tool_observation": {
                            "success": True,
                            "current_path": "/alpha",
                            "path": "/alpha/target.md",
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert parse_file_trace(trace) == frozenset(
        {"/root", "/root/alpha", "/root/alpha/target.md"}
    )


def test_normalize_file_path() -> None:
    assert _normalize_file_path("/") == "/root"
    assert _normalize_file_path("/alpha/beta") == "/root/alpha/beta"
    assert _normalize_file_path("/root/alpha") == "/root/alpha"


def test_summary_uses_same_records_for_requested_metrics() -> None:
    records = [
        TraceRecord("t1", "x", frozenset({"a", "b"})),
        TraceRecord("t2", "y", frozenset({"a"})),
        TraceRecord("t3", "z", frozenset({"a", "c"})),
        TraceRecord("t4", "w", frozenset({"b"})),
    ]
    row = summarize_coverage(
        system="test",
        candidate_label="all",
        candidates={"a", "b", "c", "d"},
        records=records,
    )
    assert row.top1_node == "a"
    assert row.top1_er == 0.75
    assert row.random1_er == (0.75 + 0.50 + 0.25 + 0.0) / 4
    assert row.nodes_er_ge_10_count == 3
    assert row.nodes_er_ge_10_ratio == 0.75
    assert row.nodes_er_ge_30_count == 2
    assert row.nodes_er_ge_30_ratio == 0.50


def test_file_candidate_sets_keep_all_user_nodes_but_writable_directories(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "file"
    env_root = run_root / "env"
    env_root.mkdir(parents=True)
    (env_root / "user_tree.json").write_text(
        json.dumps(
            {
                "name": "root",
                "children": [
                    {
                        "name": "docs",
                        "children": [
                            {"name": "readme.md", "children": [], "content": "x"},
                            {"name": "target.md", "children": [], "content": "y"},
                        ],
                    },
                    {"name": "loose.txt", "children": [], "content": "z"},
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = file_candidate_sets(run_root)
    assert candidates["All non-trivial nodes"] == {
        "/root/docs",
        "/root/docs/readme.md",
        "/root/docs/target.md",
        "/root/loose.txt",
    }
    assert candidates["Attacker-writable public nodes"] == {"/root/docs"}
