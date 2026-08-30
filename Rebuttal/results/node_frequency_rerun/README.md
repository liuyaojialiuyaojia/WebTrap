# EXP-COVER-001 supplemental empirical result

This is the original measured 16-target-per-system result. It remains retained
as supplemental empirical evidence after the column-wise 72-Browser/60-File
target-allocation upper bound became the primary coverage table.

- `table.md`: paper-facing four-row table.
- `node_frequency.json` and `node_frequency.csv`: aggregate metrics and
  protocol.
- `node_rates.csv`: empirical rate for every candidate node.
- `validation.json`: non-empty trace checks, common non-root nodes, and
  SHA-256 hashes for the aggregate JSON and table.

The corresponding 16 Browser and 16 File trajectories, target manifests,
trees, tasks, runner logs, rendered Browser site, and protocol are under
`Rebuttal/runs/node_frequency_rerun/`.

The primary result is indexed at `Rebuttal/results/coverage/` and stored under
`Rebuttal/results/node_frequency_theoretical_max/`.

Recompute these files without launching a model:

```bash
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/analyze_node_frequency.py
```
