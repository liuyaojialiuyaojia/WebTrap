# WebTrap Rebuttal experiments

This directory is the canonical implementation and evidence package for the
three reviewer-requested follow-up experiments:

1. `EXP-PLACE-001`: move Inertia and/or Payload to nearby reachable detours.
2. `EXP-COVER-001`: measure clean user-tree node exposure.
3. `EXP-ABL-001`: retain Lure, Inertia, or Payload alone.

Implementation status and result status are deliberately separate. All three
experiments are implemented. Only `EXP-COVER-001` has a completed formal run
in this directory. `EXP-PLACE-001` and `EXP-ABL-001` still require checked
injection generation and new DeepSeek v3.1 Terminal trajectories.

## Canonical layout

- `action_plan.md`: locked protocol, metrics, and current status.
- `review_matrix.md`: reviewer concern to experiment mapping.
- `experiments/`: formal planners, generators, validators, runners, and
  aggregators only.
- `tests/`: regression tests and the read-only implementation audit.
- `results/coverage/`: index for the primary and supplemental
  `EXP-COVER-001` results.
- `results/node_frequency_theoretical_max/`: primary `EXP-COVER-001`
  column-wise upper-bound table, target allocations, and Pareto frontiers.
- `results/node_frequency_rerun/`: supplemental 16+16 empirical table,
  aggregate result, per-node ER, and validation hashes.
- `results/model_ablation/`: strict One-time (`r01`) model-ablation results;
  these are kept separate from best-of-n batch summaries.
- `runs/node_frequency_rerun/`: the 32 raw clean trajectories, target
  manifests, user trees, rendered Browser site, task packages, runner logs,
  and protocol manifest for `EXP-COVER-001`.
- `results/placement/selected_placements.json`: exact structural selections
  consumed by the `EXP-PLACE-001` materializer.
- `input/`: immutable source-document hashes retained from the old server.

Generated placeholder environments for experiments 2 and 3 are intentionally
not retained. They contain no formal result and are regenerated from the
preserved source runs before checked text generation.

## Current result: EXP-COVER-001

The primary coverage result is the target-allocation upper bound for 72
Browser and 60 File trajectories. Random-1 is the expected total number of
encounters across the complete batch after uniformly sampling one candidate
node. The two threshold columns are maximized independently and are not
jointly attainable by one target allocation.

Top-k below means the kth-highest individual candidate-node ER, not the ER of
a top-k union.

| System | Candidate nodes | Top-1 ER upper bound | Top-2 ER upper bound | Top-3 ER upper bound | Random-1 expected encounters | max nodes ER ≥ 10% | max nodes ER ≥ 30% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Browser | All non-trivial nodes | 100.00% | 100.00% | 100.00% | 0.634 | 6.65% | 2.54% |
| Browser | Attacker-writable public nodes | 100.00% | 100.00% | 100.00% | 0.634 | 6.65% | 2.54% |
| File | All non-trivial nodes | 100.00% | 100.00% | 100.00% | 0.230 | 3.18% | 1.11% |
| File | Attacker-writable public nodes | 100.00% | 100.00% | 100.00% | 0.621 | 8.39% | 2.99% |

Recompute the primary result:

```bash
PYTHONPATH=. python3 \
  Rebuttal/experiments/optimize_node_frequency_targets.py
```

The original empirical result is retained as supplemental evidence. Browser
and File each ran 16 fresh, clean trajectories against 16 distinct
prefix-balanced target leaves:

| System | Candidate nodes | Top-1 ER | Random-1 ER | nodes ER ≥ 10% | nodes ER ≥ 30% |
| --- | --- | ---: | ---: | ---: | ---: |
| Browser | All non-trivial nodes | 100.00% | 0.78% | 1.57% | 0.88% |
| Browser | Attacker-writable public nodes | 100.00% | 0.78% | 1.57% | 0.88% |
| File | All non-trivial nodes | 100.00% | 0.48% | 0.84% | 0.58% |
| File | Attacker-writable public nodes | 100.00% | 1.08% | 1.95% | 1.26% |

Recompute the supplemental empirical table from retained trajectories:

```bash
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/analyze_node_frequency.py
```

The complete 16-target rerun entry point, which does make new model calls, is:

```bash
bash Rebuttal/experiments/run_node_frequency_rerun.sh
```

## Experiments 2 and 3

Rebuild the deterministic placement plan:

```bash
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/structural_variants.py
```

The materializer defines the three single-stage variants directly from the
locked Full-stage locations; no separate single-stage manifest is required.
Execute the formal pipeline in this order:

```bash
# No model calls; creates Rebuttal-local placeholder environments.
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/materialize_formal_envs.py

# Makes attacker/checker model calls; only checker-approved environments
# become runnable.
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/generate_checked_injections.py

# Refuses environments that do not pass the formal contract.
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/run_formal_deepseek.py --target all

# Admits only complete, contract-valid outputs into the summary.
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/experiments/summarize_rebuttal_results.py
```

The formal contract locks the target/checker models, temperature, seeds,
trial/sample counts, action/step budgets, defense mode, route semantics,
retained stages, expected nodes, environment fingerprint, and evaluator
counts. Copied Full-condition text, checker rejection, incomplete cases, and
stale or mismatched outputs cannot be reported as formal results.

## Verification

```bash
PYTHONPATH=. wasp/visualwebarena/venv/bin/python \
  Rebuttal/tests/audit_implementation.py

PYTHONPATH=. wasp/visualwebarena/venv/bin/pytest \
  Rebuttal/tests \
  web/psaa/ablation/test_inject_from_attack_case.py
```
