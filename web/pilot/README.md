# Motivation Detour Experiment

`web/pilot/` contains the motivation experiment for mid-task benign detours in web navigation.

## Main Command

```bash
python web/pilot/run_experiment.py --trials 1 --port 8010 --model <model>
```

## Main Scripts

| Script | Purpose |
| --- | --- |
| `build_cases.py` | Prepare reusable pilot cases from generated web snapshots. |
| `materialize_sites.py` | Render pilot site variants. |
| `sample_histories.py` | Sample shared histories that reach the anchor page. |
| `run_warm_start_trials.py` | Continue from the anchor page under detour conditions. |
| `classify_trials.py` | Label trials as `Unaffected`, `Hijacked`, or `Perturbed`. |
| `summarize_results.py` | Summarize pilot results. |

## Parameters

| Argument | Meaning |
| --- | --- |
| `--model` | Agent model. |
| `--trials` | Post-anchor trials per condition. |
| `--sampling-trials` | Shared-history sampling attempts. |
| `--sampling-max-actions` | Action cap for sampling. |
| `--max-actions` | Action cap for post-anchor continuation. |
| `--concurrency` | Worker count. |
| `--port` | Static site port. |
