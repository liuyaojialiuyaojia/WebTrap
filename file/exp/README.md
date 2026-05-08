# File-Browser Experiment Pipeline

`file/exp/` runs the file-browser experiments in the WebTrap paper. It builds task cases from a released source tree, applies attacks, runs the file agent, and evaluates the traces.

## Main Command

```bash
RUN_ID=<run_id> MODEL=<model> bash file/exp/run_batch.sh
```

Outputs are written to `file/runs/<run_id>/`.

## Pipeline Stages

| Stage | Script | Purpose |
| --- | --- | --- |
| `exp1` | `exp1/run_exp1.sh` | Build user/security trees and write `attack_cases.jsonl`. |
| `attack` | `attack/run_attack.sh` | Inject `baseline`, `psaa`, `topicattack`, or `combinedattack`. |
| `exp2` | `exp2/run_exp2.sh` | Run the tool-call file agent. |
| `exp3` | `exp3/run_exp3.sh` | Evaluate user goal, attacker goal, and ASR-intermediate. |

## Common Parameters

| Variable | Meaning |
| --- | --- |
| `RUN_ID` | Run directory name. |
| `RUNS_ROOT` | Output root, default `file/runs`. |
| `SOURCE_TREE` | Source tree JSON, default `file/create/file_tree/custom_d11_w2_r_1.json`. |
| `TOP_K` | Number of cases. |
| `DEPTH` / `WIDTH` | User subtree shape. |
| `ANCHOR_LOGICAL_PATH` | Security subtree insertion path. |
| `INJECTION_MODE` | `baseline`, `psaa`, `topicattack`, or `combinedattack`. |
| `MODEL` | Agent model. |
| `USER_QUESTION_MODEL` | User-question generation model. |
| `ASR_MODEL` | ASR-intermediate judge model. |
| `MAX_STEPS` | Agent action budget. |
| `NUM_SAMPLES` | Independent samples per case. |
| `DEFENSE_MODE` | Defense wrapper used by the agent. |

## Stage Entrypoints

```bash
bash file/exp/exp1/run_exp1.sh
INJECTION_MODE=psaa bash file/exp/attack/run_attack.sh --run-dir file/runs/<run_id> --case-id case_0001
bash file/exp/exp2/run_exp2.sh --run-dir file/runs/<run_id> --case-id case_0001
bash file/exp/exp3/run_exp3.sh --run-dir file/runs/<run_id>
```
