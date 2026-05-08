# Web-Browser Experiment Pipeline

`web/exp/` builds the web-browser environments, generates WebArena-compatible tasks, runs browser agents, and evaluates WebTrap or baseline attacks.

## Main Commands

Run WebTrap:

```bash
EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=<run_id> MODEL=<model> \
  bash web/exp/run_batch_psaa_attacks.sh
```

Run web baselines:

```bash
EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=<run_id> MODEL=<model> \
  bash web/exp/run_batch_baseline_attacks.sh
```

Run OSR-only evaluation:

```bash
EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=<run_id> MODEL=<model> \
  bash web/exp/osr/run_osr.sh
```

## Pipeline Stages

| Stage | Scripts | Purpose |
| --- | --- | --- |
| Stage 02 | `02_nested_builder/extract_subtree.py` | Extract a navigation subtree from the released seed tree. |
| Stage 02.5 | `02_5_security_microtree/compile_security_microtree.py` | Insert GitLab or Reddit WASP-derived security microtrees. |
| Stage 03 | `03_a2perf_render/pack_static_site.py` | Package the site into static files for agent execution. |
| Stage 04 | `04_task_gen/*.py` | Generate user tasks and aligned attacker tasks. |
| Stage 05 | `05_evaluate/run_full.sh` | Run the agent, extract paths, and write metrics. |

## Common Parameters

| Variable | Meaning |
| --- | --- |
| `EXPERIMENT_ROOT` | Output root, default `web/runs`. |
| `EXPERIMENT_ID` | Run directory under `EXPERIMENT_ROOT`. |
| `SOURCE_ROOT` | Seed tree directory, default `web/exp/02_nested_builder/tree/d10_w2_s42`. |
| `DEPTH` / `WIDTH` | Extracted user-area tree shape. |
| `WASP_ENV` | `gitlab` or `reddit`. |
| `WASP_EVAL_TYPE` | `exfil`, `program_html`, or `both`. |
| `SECURITY_SPEC` | Security microtree spec. Defaults to the spec matching `WASP_ENV`. |
| `ANCHOR_BREADCRUMB` | Page breadcrumb where the security hub is attached. |
| `USER_TASK_COUNT` | Number of user tasks generated from leaf pages. |
| `TRIALS` | Agent trials per task. |
| `MODEL` | Browser-agent model. |
| `AGENT_DEFENSE_MODE` | Defense wrapper used during Stage 05. |

Additional WebTrap variables:

| Variable | Meaning |
| --- | --- |
| `PSAA_VARIANT` | `full`, `wo_lure`, `wo_inertia`, or `wo_payload`. |
| `PSAA_PROMPT_SPEC` | Optional YAML prompt-spec override. |
| `INJECT_START_PAGE_PATH` | Injection start page for dynamic WebTrap injection. |

Additional baseline variables:

| Variable | Meaning |
| --- | --- |
| `BASELINE_FORMATS` | Comma-separated baseline list. |
| `INJECT_PAGE_PATH` | Page path for single-shot baseline injection. |
| `TOPICATTACK_MODEL` | Generator model for TopicAttack. |
| `TOPICATTACK_NUM_TURNS` | Number of TopicAttack turns. |
| `COMBINEDATTACK_SEED` | CombinedAttack sampling seed. |

## Stage-Level Entrypoints

```bash
python web/exp/02_nested_builder/extract_subtree.py --help
python web/exp/02_5_security_microtree/compile_security_microtree.py --help
python web/exp/03_a2perf_render/pack_static_site.py --help
python web/exp/04_task_gen/make_user_tasks_from_leaves.py --help
python web/exp/04_task_gen/make_wasp_attack_tasks_from_manifest.py --help
bash web/exp/05_evaluate/run_full.sh --with-attacker
```
