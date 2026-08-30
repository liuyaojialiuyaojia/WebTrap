from Rebuttal.experiments.optimize_node_frequency_targets import (
    TreeNode,
    allocation_metrics,
    maximize_ranked_er_upper_bounds,
    maximize_threshold,
    threshold_pareto_frontier,
)


def test_two_leaf_tree_threshold_optimization() -> None:
    first = TreeNode("first", targetable=True)
    second = TreeNode("second", targetable=True)
    root = TreeNode("root", children=[first, second], is_directory=True)
    policy = lambda node: node is not root

    score, allocation = maximize_threshold(
        root,
        trajectories=4,
        minimum_visits=2,
        is_candidate=policy,
    )
    metrics = allocation_metrics(
        root,
        allocation=allocation,
        trajectories=4,
        minimum_visits_10=1,
        minimum_visits_30=2,
        is_candidate=policy,
    )

    assert score == 2
    assert sorted(allocation.values()) == [2, 2]
    assert metrics["top1_er"] == 0.5
    assert metrics["top2_er"] == 0.5
    assert metrics["top3_er"] == 0.0
    assert metrics["random1_expected_encounters"] == 2.0
    assert metrics["nodes_er_ge_30_count"] == 2


def test_ranked_er_upper_bounds_use_kth_highest_individual_node() -> None:
    left_leaf = TreeNode("left_leaf", targetable=True)
    right_leaf = TreeNode("right_leaf", targetable=True)
    left = TreeNode(
        "left",
        children=[left_leaf],
        is_directory=True,
    )
    right = TreeNode(
        "right",
        children=[right_leaf],
        is_directory=True,
    )
    root = TreeNode("root", children=[left, right], is_directory=True)
    policy = lambda node: node is not root

    maxima = maximize_ranked_er_upper_bounds(
        root,
        trajectories=4,
        ranks=(1, 2, 3),
        is_candidate=policy,
    )

    assert {rank: result[0] for rank, result in maxima.items()} == {
        1: 4,
        2: 4,
        3: 2,
    }
    assert sorted(maxima[3][1].values()) == [2, 2]


def test_threshold_frontier_detects_tradeoff() -> None:
    leaves = [
        TreeNode(f"leaf_{index}", targetable=True)
        for index in range(3)
    ]
    root = TreeNode("root", children=leaves, is_directory=True)
    policy = lambda node: node is not root

    frontier = threshold_pareto_frontier(
        root,
        trajectories=6,
        minimum_visits_10=1,
        minimum_visits_30=3,
        is_candidate=policy,
    )

    assert frontier == ((3, 1), (2, 2))
