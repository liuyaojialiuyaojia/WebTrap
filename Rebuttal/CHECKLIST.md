# Rebuttal status

## EXP-COVER-001 — complete

- [x] Primary 72-Browser/60-File target-allocation upper-bound analysis.
- [x] Random-1 reported as expected total encounters across the full batch.
- [x] Independent ER ≥ 10% and ER ≥ 30% maxima and Pareto frontiers retained.
- [x] Primary result explicitly marked theoretical and column-wise.
- [x] 16 fresh clean Browser trajectories.
- [x] 16 fresh clean File trajectories.
- [x] 16 distinct prefix-balanced target leaves per system.
- [x] Root excluded, targets retained, one count per node per trajectory.
- [x] Supplemental empirical four-row table and per-trajectory Random-1 ER.
- [x] Original raw traces, protocol, aggregate result, per-node ER, and hashes
  retained.

## EXP-PLACE-001 — implemented, not run

- [x] `shift_s2`, `shift_s3`, and `shift_s2s3` structural selections.
- [x] Browser and File materializer.
- [x] Route-specific attacker/checker generation.
- [x] Environment and runtime contract validation.
- [x] Formal DeepSeek runner and result aggregation.
- [ ] Regenerate formal environments.
- [ ] Generate checker-approved injection text.
- [ ] Run formal Browser and File trajectories/evaluation.

## EXP-ABL-001 — implemented, not run

- [x] `lure_only`, `inertia_only`, and `payload_only` plans at Full locations.
- [x] Standalone route semantics for a retained non-Payload stage.
- [x] Browser-GitLab and supplementary File materializer.
- [x] Checked generation, contract validation, formal runner, and aggregation.
- [ ] Regenerate formal environments.
- [ ] Generate checker-approved injection text.
- [ ] Run formal Browser and File trajectories/evaluation.

Only checked, complete, contract-valid outputs may change either pending
experiment to complete.
