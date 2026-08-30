# Rebuttal review matrix

| Item | Reviewer concern | Class | Route | Experiment | Status |
| --- | --- | --- | --- | --- | --- |
| R-qziR-C2 | Exact shortest-path placement may require unrealistic topology knowledge. | experiment gap / claim scope | supplementary experiment | EXP-PLACE-001 | all three formal variants implemented; checked generation/model run pending |
| R-QCsZ-C1 | Initial lure exposure is standardized at the root; quantify task-independent high-exposure locations. | experiment gap | target-allocation upper bound plus supplementary clean rerun | EXP-COVER-001 | completed: primary 72/60 column-wise coverage upper bounds and Pareto frontier; original 16+16 empirical rerun retained |
| R-QCsZ-C2 | Leave-one-out ablation does not directly show whether any single stage is sufficient. | experiment gap | component ablation | EXP-ABL-001 | single-stage formal pipeline implemented; checked generation/model run pending |
| R-qziR-C3 | Explain non-monotonic UUA and recovery under adjusted budgets. | experiment gap | deferred by user | — | deferred |
| R-QCsZ-C3 | Evaluate recent GPT and Claude models. | experiment gap | deferred by user | — | deferred |

## Evidence boundary

The primary EXP-COVER-001 table is a column-wise theoretical upper bound under
ideal root-to-target navigation; its 10% and 30% maxima do not describe one
jointly attainable run. The retained 16+16 rerun is empirical supplemental
evidence. Neither result claims that every deployment task encounters a lure.
EXP-PLACE-001 tests a minimal reachable deviation outside the original
shortest path; it does not establish robustness to arbitrary placement error.
Under the current safety constraint, EXP-PLACE-001 and EXP-ABL-001 have not
launched new model-driven attack trajectories. Their retained artifacts are
the structural placement selection, formal pipeline code, locked contracts,
and regression tests. True shifted-placement ASR and true single-stage ASR
still require controlled model reruns.
