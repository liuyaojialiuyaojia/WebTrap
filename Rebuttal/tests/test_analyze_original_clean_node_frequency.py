from pathlib import Path

from Rebuttal.experiments.analyze_original_clean_node_frequency import (
    original_file_candidate_sets,
)


def test_original_file_candidates_exclude_security_subtree(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "file_osr"
    env_root = run_root / "env"
    env_root.mkdir(parents=True)
    env_root.joinpath("env_pre_injection_tree.json").write_text(
        """
        {
          "name": "root",
          "children": [
            {
              "name": "docs",
              "children": [
                {"name": "guide.md", "children": []},
                {
                  "name": "security_entry",
                  "children": [
                    {"name": "payload.txt", "children": []}
                  ]
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    run_root.joinpath("attack_cases.jsonl").write_text(
        (
            '{"security_entry_logical_path": '
            '"/root/docs/security_entry"}\n'
        ),
        encoding="utf-8",
    )

    candidates, prefixes = original_file_candidate_sets(run_root)

    assert prefixes == {"/root/docs/security_entry"}
    assert candidates["All non-trivial nodes"] == {
        "/root/docs",
        "/root/docs/guide.md",
    }
    assert candidates["Attacker-writable public nodes"] == {"/root/docs"}
