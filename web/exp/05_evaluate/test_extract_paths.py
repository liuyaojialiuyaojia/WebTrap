import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "extract_paths.py"
SPEC = importlib.util.spec_from_file_location("extract_paths_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extract_paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract_paths)


def _write_trace(tmp_path: Path, messages: list[dict]) -> Path:
    path = tmp_path / "trace_1000.jsonl"
    path.write_text(json.dumps(messages) + "\n", encoding="utf-8")
    return path


def test_parse_tool_use_log_handles_wrapped_single_item_click_args(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "goto",
                            "arguments": "{\"url\": \"http://127.0.0.1:8000/index.html\"}",
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "OBSERVATION:\nURL: http://127.0.0.1:8000/index.html\n",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "click",
                            "arguments": "[{\"element_id\":\"289\"}]",
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "ERROR: Tool arguments must be a JSON object.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "click",
                            "arguments": "{\"element_id\": \"289\"}",
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "OBSERVATION:\n"
                    "Information navigation → page 6\n"
                    "URL: http://127.0.0.1:8000/index.html#p6\n"
                ),
            },
        ],
    )

    data = extract_paths._parse_tool_use_log(
        trace_path,
        {"289": {"target_breadcrumb": "/1/1"}},
        {
            0: {"breadcrumb": "/"},
            6: {"breadcrumb": "/1/1"},
        },
        interpolate_ancestors=False,
    )

    assert data["breadcrumbs"] == ["/", "/1/1"]
    assert data["actions"][1]["argument_warning"] == "wrapped_single_item_list"
    assert data["click_sequence"] == [
        {
            "source": "/",
            "target": "/1/1",
            "type": "click",
            "via": "289",
            "source_page": 0,
            "expected_target": "/1/1",
        }
    ]


def test_same_page_observation_clears_stale_pending_click(tmp_path: Path) -> None:
    trace_path = _write_trace(
        tmp_path,
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "click",
                            "arguments": "{\"element_id\": \"100\"}",
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": "OBSERVATION:\nURL: http://127.0.0.1:8000/index.html\n",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "click",
                            "arguments": "{\"element_id\": \"200\"}",
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "content": (
                    "OBSERVATION:\n"
                    "Information navigation → page 2\n"
                    "URL: http://127.0.0.1:8000/index.html#p2\n"
                ),
            },
        ],
    )

    data = extract_paths._parse_tool_use_log(
        trace_path,
        {
            "100": {"target_breadcrumb": "/stale"},
            "200": {"target_breadcrumb": "/1"},
        },
        {
            0: {"breadcrumb": "/"},
            2: {"breadcrumb": "/1"},
        },
        interpolate_ancestors=False,
    )

    assert data["click_sequence"] == [
        {
            "source": "/",
            "target": "/1",
            "type": "click",
            "via": "200",
            "source_page": 0,
            "expected_target": "/1",
        }
    ]
