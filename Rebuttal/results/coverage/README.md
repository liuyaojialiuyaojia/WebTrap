# EXP-COVER-001 result index

## Primary result

The paper-facing coverage result is the column-wise theoretical
target-allocation upper bound for 72 Browser and 60 File trajectories:

- table: `../node_frequency_theoretical_max/table.md`
- exact rows: `../node_frequency_theoretical_max/theoretical_max.json`
- independently optimized target assignments:
  `../node_frequency_theoretical_max/target_allocations.json`
- jointly attainable tradeoffs:
  `../node_frequency_theoretical_max/pareto_frontiers.json`

Random-1 is the expected total encounter count across the complete batch after
uniformly sampling one candidate node. The ER >= 10% and ER >= 30% columns are
maximized independently; one target allocation cannot attain both column
maxima. Top-k is the kth-highest individual candidate-node ER rather than a
top-k union. The Top-1, Top-2, and Top-3 upper bounds are all 100% and are
jointly attained by concentrating the batch on the retained deep target.

## Supplemental empirical result

The original 16-target-per-system clean rerun remains complete and unchanged:

- table: `../node_frequency_rerun/table.md`
- aggregate: `../node_frequency_rerun/node_frequency.json`
- per-node rates: `../node_frequency_rerun/node_rates.csv`
- validation hashes: `../node_frequency_rerun/validation.json`
- raw trajectories and protocol: `../../runs/node_frequency_rerun/`

The pre-existing concentrated 72-Browser/60-File OSR measurement is retained
separately under `../node_frequency_original_clean/` as diagnostic evidence;
it is not the primary upper-bound table.
