# Web Baselines

`web/baseline/` contains the web-browser attack baselines used in the paper.

## Baselines

- WASP formats:
  - `goal_hijacking_url_injection`
  - `goal_hijacking_plain_text`
  - `generic_url_injection`
  - `generic_plain_text`
- `topicattack`
- `combinedattack`

## Main Command

```bash
EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=<run_id> \
BASELINE_FORMATS=goal_hijacking_plain_text,topicattack \
bash web/exp/run_batch_baseline_attacks.sh
```

## Parameters

| Variable | Meaning |
| --- | --- |
| `BASELINE_FORMATS` | Comma-separated baseline list. |
| `ATTACK_CASE` | Attack-case JSON for single-run injection scripts. |
| `INJECT_PAGE_PATH` | Page path for single-shot injection. |
| `BASELINE_USER_TASK_PATH` | User-task JSON for goal-hijacking baselines. |
| `TOPICATTACK_MODEL` | TopicAttack generator model. |
| `TOPICATTACK_NUM_TURNS` | TopicAttack turn count. |
| `COMBINEDATTACK_SEED` | CombinedAttack sampling seed. |

## Direct Entrypoints

```bash
bash web/baseline/run_injection.sh
bash web/baseline/topicattack/run_injection.sh
bash web/baseline/combinedattack/run_injection.sh
```
