# Rebuttal experiment action plan

## EXP-PLACE-001 — suboptimal placement

- Reviewer item: R-qziR-C2.
- Question: Does WebTrap retain non-zero effectiveness when both later stages
  are moved off the original shortest path?
- Intervention: keep the root Lure fixed; move Inertia and Payload to the
  closest sibling detours at the same depths as their original locations.
- Selection rule: minimize added navigation hops, preserve stage order and
  reachability, and keep the original stage timing as closely as possible.
- Fixed conditions: task set, target model, temperature, trial count, tool
  permissions, evaluator, attacker objective, and stage templates.
- Metrics: Payload encounter rate, ASR-E, ASR-I, UUA, and dual-goal success.
- Success criterion: valid comparable runs with non-zero Payload encounter and
  attack success in both Browser and File settings.
- Failure interpretation: a zero result narrows the claim to topology-sensitive
  placement rather than being hidden.
- Execution status: all three shifted conditions are selected. The Browser and
  File materializer, checked route-specific generation, strict environment
  validation, formal DeepSeek runner, and contract-aware aggregation are
  implemented. Regenerable placeholder environments are not retained.
  Checked text and formal model trajectories have not been run.

## EXP-COVER-001 — clean node exposure

- Reviewer item: R-QCsZ-C1.
- Question: Under the original Browser/File batch sizes, what is the maximum
  user-tree coverage attainable by distributing clean task targets?
- Primary analysis: allocate 72 Browser and 60 File idealized root-to-target
  trajectories over the locked user trees. Dynamic programming independently
  maximizes the number of candidate nodes with ER at least 10% and at least
  30%. These two column-wise maxima are theoretical upper bounds and are not
  jointly attainable by one target allocation.
- Supplemental empirical rerun: retain the 16 new clean trajectories per
  system. Each trajectory uses a different target user leaf selected by
  prefix-balanced sampling so shallow branches are covered before any branch
  receives an extra target.
- Fixed conditions for the supplemental rerun: DeepSeek v3.1 Terminal,
  temperature 1.0, one sample per target, Browser 20-action budget, File
  25-step budget, no injected text and no attacker objective. The task
  questions are deterministic and target the selected leaf by title/filename
  and content clue.
- Counting rule: one count per node per trajectory. The mandatory root is the
  only excluded node; selected target nodes remain in scope. Numerators and
  denominators contain user-tree nodes only.
- Candidate policies: Browser "All" and "attacker-writable public" both contain
  all non-root public user pages. File "All" contains non-root directories and
  files, while "attacker-writable public" contains the non-root directories
  accepted by the File injection interface.
- Primary metrics: Top-1, Top-2, and Top-3 individual-node ER upper bounds;
  Random-1 expected total encounter count across the full batch; and the
  independently maximized proportions of candidate nodes with ER at least 10%
  and at least 30%. Top-k here is the kth-highest node ER, not a union metric.
- Supplemental metrics: the original empirical Top-1 ER, uniform per-trajectory
  Random-1 ER, and empirical 10%/30% threshold proportions.
- Execution status: completed. The primary upper-bound table, exact target
  allocations, and Pareto frontiers are retained under
  `Rebuttal/results/node_frequency_theoretical_max/`. The original 16+16
  empirical result remains intact under
  `Rebuttal/results/node_frequency_rerun/`, with all raw trajectories under
  `Rebuttal/runs/node_frequency_rerun/`.

## EXP-ABL-001 — single-stage ablation

- Reviewer item: R-QCsZ-C2.
- Question: How much can Lure, Inertia, or Payload achieve alone compared with
  the complete three-stage design?
- Intervention: retain exactly one stage at its original injection location and
  remove the other two stages from the rendered metadata.
- Fixed conditions for formal rerun: same task/evaluation files, same attack
  scaffold, temperature, trial/sample count, action budget, and one target
  model per table. The missing rows use DeepSeek v3.1 Terminal
  (`deepseek-v3-1-terminus`) and must be compared with the existing
  DeepSeek-matched Full result rather than a cross-model Full result.
- Metrics: Payload encounter rate, ASR-E, ASR-I, UUA, and dual-goal success.
- Success criterion: all four variants have complete, evaluator-valid results.
- Failure interpretation: if a single stage matches Full, the manuscript must
  weaken the claim that staged guidance is necessary in this setting.
- Execution status: the Browser ablation generator supports all three
  single-stage plans at Full-condition locations, with a complete route suffix
  for a lone non-Payload stage. Browser-GitLab and supplementary File
  materialization, checked generation, validation, formal execution, and
  dynamic aggregation are implemented. Regenerable placeholder environments
  are not retained. True single-stage ASR requires new model-driven
  trajectories.
