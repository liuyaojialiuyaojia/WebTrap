from pathlib import Path

from Rebuttal.experiments.structural_variants import (
    Graph,
    _rank_candidates,
    load_browser_graph,
    load_file_graph,
)


def test_rank_candidates_moves_both_stages_off_original_path() -> None:
    adjacency = {
        0: {1, 4},
        1: {0, 2},
        2: {1, 3, 5},
        3: {2},
        4: {0, 5},
        5: {4, 2},
    }
    graph = Graph(adjacency, {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2})
    rows = _rank_candidates(
        system="Browser",
        graph=graph,
        start=0,
        anchor=3,
        original_inertia=1,
        original_payload=2,
        candidates=set(adjacency),
    )
    assert rows
    assert rows[0].inertia_node == 4
    assert rows[0].payload_node == 5
    assert rows[0].route == (0, 4, 5, 2, 3)
    assert rows[0].added_hops == 1
    assert rows[0].moved_stages == ("inertia", "payload")
    assert rows[0].inertia_displacement_hops == 2
    assert rows[0].payload_displacement_hops == 1


def test_rank_candidates_supports_each_planned_shift_condition() -> None:
    adjacency = {
        0: {1, 4},
        1: {0, 2},
        2: {1, 3, 5},
        3: {2},
        4: {0, 5},
        5: {4, 2},
    }
    graph = Graph(adjacency, {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 2})

    shift_s2 = _rank_candidates(
        system="Browser",
        graph=graph,
        start=0,
        anchor=3,
        original_inertia=1,
        original_payload=2,
        candidates=set(adjacency),
        moved_stages=("inertia",),
        variant="shift_s2",
    )[0]
    shift_s3 = _rank_candidates(
        system="Browser",
        graph=graph,
        start=0,
        anchor=3,
        original_inertia=1,
        original_payload=2,
        candidates=set(adjacency),
        moved_stages=("payload",),
        variant="shift_s3",
    )[0]

    assert shift_s2.variant == "shift_s2"
    assert shift_s2.inertia_node == 4
    assert shift_s2.payload_node == 2
    assert shift_s2.inertia_displacement_hops == 2
    assert shift_s2.payload_displacement_hops == 0
    assert shift_s3.variant == "shift_s3"
    assert shift_s3.inertia_node == 1
    assert shift_s3.payload_node == 5
    assert shift_s3.inertia_displacement_hops == 0
    assert shift_s3.payload_displacement_hops == 1


def test_browser_loader_ignores_non_structural_fields(tmp_path: Path) -> None:
    path = tmp_path / "pages.json"
    path.write_text(
        """
        {
          "pages": [
            {
              "page_index": 0,
              "path": [],
              "body": "opaque",
              "click_targets": [{"target_page": 1, "label": "opaque"}]
            },
            {
              "page_index": 1,
              "path": [0],
              "injections": [{"text": "opaque"}],
              "click_targets": [{"target_page": 0}]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    graph = load_browser_graph(path)
    assert graph.depths == {0: 0, 1: 1}
    assert graph.shortest_path(0, 1) == (0, 1)


def test_file_loader_keeps_directories_only(tmp_path: Path) -> None:
    path = tmp_path / "tree.json"
    path.write_text(
        """
        {
          "name": "root",
          "type": "directory",
          "children": [
            {
              "name": "alpha",
              "type": "directory",
              "content": "opaque",
              "children": [
                {
                  "name": "nested",
                  "type": "directory",
                  "children": [
                    {"name": "note.txt", "type": "file", "content": "opaque"}
                  ]
                },
                {"name": "loose.txt", "type": "file", "content": "opaque"}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    graph = load_file_graph(path)
    assert graph.depths == {
        "/root": 0,
        "/root/alpha": 1,
        "/root/alpha/nested": 2,
    }
    assert graph.shortest_path("/root", "/root/alpha") == (
        "/root",
        "/root/alpha",
    )
