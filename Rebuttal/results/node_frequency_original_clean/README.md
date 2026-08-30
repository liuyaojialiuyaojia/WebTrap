# Original clean-run node-frequency result

This table reuses the pre-existing clean OSR runs rather than the fresh
prefix-balanced 16-target rerun:

- Browser: `web/runs/exp_d10w2_osr_gitlab`, 24 tasks × 3 trials = 72
  trajectories, with 2 unique user target leaves.
- File: `file/runs/exp_d10w2_osr`, 20 cases × 3 samples = 60 trajectories,
  with 1 unique user target file.

The metric definitions match `results/node_frequency_rerun`: each node counts
at most once per trajectory, the mandatory root is excluded, and only
user-tree nodes are included in the numerator and denominator. Neutral
security-microtree nodes present in the clean OSR environments are therefore
excluded.

Recompute without model calls:

```bash
PYTHONPATH=. python3 \
  Rebuttal/experiments/analyze_original_clean_node_frequency.py
```

The paper-facing result is in `table.md`; exact counts and rates are in
`node_frequency.json`, `node_frequency.csv`, and `node_rates.csv`.
