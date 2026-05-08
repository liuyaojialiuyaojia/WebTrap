from file.exp.attack import common as attack_common


def test_ensure_readme_and_append_injects_raw_text_with_single_newline_separator() -> None:
    tree = {
        "name": "root",
        "children": [
            {
                "name": "docs",
                "children": [
                    {"name": "readme.md", "content": "Original body", "children": []}
                ],
            }
        ],
    }

    readme_logical_path, before_size, after_size = attack_common.ensure_readme_and_append(
        tree,
        "/root/docs",
        root_logical="/root",
        marker="ignored",
        content="Injected body",
    )

    assert readme_logical_path == "/root/docs/readme.md"
    assert before_size == len("Original body")
    assert after_size == len("Original body\nInjected body\n")
    assert tree["children"][0]["children"][0]["content"] == (
        "Original body\nInjected body\n"
    )
