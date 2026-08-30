# Theoretical target-allocation maxima

This directory contains theoretical upper bounds, not measured trajectories.
It answers how the 72 Browser trajectories and 60 File trajectories could be
assigned to user targets to maximize the two threshold columns.

## Random-1 definition

`Random-1 expected encounters (batch)` is the expected total number of
encounters across the complete batch after sampling one candidate node
uniformly:

```text
sum(candidate-node visit counts) / candidate-node count
```

It is a count, not a percentage and not a per-trajectory encounter rate.

## Ranked ER upper bounds

Top-k is the kth-highest individual candidate-node ER, not the encounter rate
of the union of the top k nodes. Each rank is optimized over target
allocations. For all four reported candidate policies, Top-1, Top-2, and Top-3
are each 100%. These three maxima are jointly attainable here: assigning every
trajectory to the retained deep target makes at least three candidate nodes
on its root-to-target path appear in every trajectory.

`target_allocations.json` retains a witness allocation for each ranked upper
bound as well as for the two threshold objectives.

## Threshold optimization

Each idealized trajectory visits every user-tree node on the unique path from
the root to its assigned target exactly once. The mandatory root is excluded.

- Browser ER >= 10% requires at least 8 of 72 trajectories. The maximum uses
  9 deep, prefix-balanced targets with 8 trajectories each.
- Browser ER >= 30% requires at least 22 trajectories. The maximum uses
  3 prefix-balanced targets with allocations 28, 22, and 22.
- File ER >= 10% requires at least 6 of 60 trajectories. The maximum uses
  10 deep, prefix-balanced targets with 6 trajectories each.
- File ER >= 30% requires at least 18 trajectories. The maximum uses
  3 prefix-balanced targets with allocations 24, 18, and 18.

The 10% and 30% maxima are not jointly attainable by one allocation. Therefore
`table.md` is a column-wise upper-bound table. It must not be described as one
empirical run or one simultaneously attainable target design.

`pareto_frontiers.json` contains all nondominated, jointly attainable
threshold-count pairs. `target_allocations.json` contains example allocations
that independently maximize every reported ranked or threshold objective.

Recompute:

```bash
PYTHONPATH=. python3 \
  Rebuttal/experiments/optimize_node_frequency_targets.py
```
