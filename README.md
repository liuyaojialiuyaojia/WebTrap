<div align="center">

# 🕸️ WebTrap

**[Stealthy Mid-Task Hijacking of Browser Agents During Navigation](https://arxiv.org/abs/2605.08310)**

[TL;DR](#tldr) • [Overview](#overview) • [Code Map](#code-map) • [Setup](#setup) • [Run Experiments](#run-experiments) • [Citation](#citation)

[[Paper](https://arxiv.org/abs/2605.08310)] [[PDF](https://arxiv.org/pdf/2605.08310)]

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status: research code](https://img.shields.io/badge/status-research%20code-lightgrey)
![Topic: Agent Security](https://img.shields.io/badge/topic-agent%20security-purple)
![Task: Browser Agents](https://img.shields.io/badge/task-browser%20agents-orange)

</div>

<p align="center">
  <img src="./materials/fig1.png" alt="WebTrap motivation: mid-task hijacking in web and file-browser tasks" width="100%">
</p>

<p align="center"><em>Motivation: a browser agent can be hijacked mid-task, complete an attacker-controlled detour, and still return to the original user workflow.</em></p>

<a id="tldr"></a>
## ✨ TL;DR

**WebTrap** studies a stealthy prompt-injection risk for long-horizon browser agents: an attacker can hijack the agent **in the middle of navigation**, steer it through an attacker-controlled prerequisite step, and then let it resume the original user task.

| What to know | WebTrap in one line |
| --- | --- |
| 🧭 Attack surface | Mid-task navigation in web and file-browser agent workflows. |
| 🪤 Core mechanism | Lure, inertia, and payload traps bind the attacker goal into the user task. |
| 🧪 Evaluation settings | Web-browser tasks and file-browser tasks. |
| 🎯 Experiment goal | Measure attack success while preserving the original task utility. |

This repository contains the experiment code used to construct tasks, inject WebTrap/baseline attacks, run agents, and evaluate outcomes.

<a id="overview"></a>
## 🔍 Overview

This repository provides the experiment code for **[WebTrap: Stealthy Mid-Task Hijacking of Browser Agents During Navigation](https://arxiv.org/abs/2605.08310)**.

WebTrap is a prompt-injection attack for long-horizon browser agents. In this work, browser tasks include both webpage navigation and file-system navigation. WebTrap uses **multi-step instruction fusion steering** and **context-grounded enhancement** to make an agent execute the attacker goal and then resume the original user task, preserving user-task utility while increasing attack success.

<p align="center">
  <img src="./materials/webtrap.png" alt="WebTrap method overview: lure, inertia, and payload traps" width="100%">
</p>

<p align="center"><em>Method overview: WebTrap uses lure, inertia, and payload traps to bind the attacker goal into the user task as a prerequisite workflow step.</em></p>

The code covers the two experimental settings in the paper:

| Setting | Base benchmark | Code |
| --- | --- | --- |
| Web browser tasks | Extended WASP environments | [`web/`](./web/) |
| File browser tasks | Extended InjecAgent-style file tasks | [`file/`](./file/) |

<a id="code-map"></a>
## 🗺️ Code Map

Core folders:

| Folder | Purpose |
| --- | --- |
| [`web/`](./web/) | Web-browser experiments, including WebTrap injection, baseline attacks, task construction, agent execution, and evaluation. |
| [`file/`](./file/) | File-browser experiments, including released file-tree data, attack construction, execution, and evaluation. |
| [`utils/`](./utils/) | Shared experiment utilities, model client wrapper, and defense baselines. |
| [`materials/`](./materials/) | README figures. |

Referenced submodules:

| Submodule | Repository | Used for |
| --- | --- | --- |
| [`wasp/`](./wasp/) | [`liuyaojialiuyaojia/wasp`](https://github.com/liuyaojialiuyaojia/wasp) | Extended WASP web environments used by the web-browser experiments. |
| [`A2Perf/`](./A2Perf/) | [`liuyaojialiuyaojia/A2Perf`](https://github.com/liuyaojialiuyaojia/A2Perf) | Browser-agent execution and evaluation components used by the web runs. |
| [`InjecAgent/`](./InjecAgent/) | [`uiuc-kang-lab/InjecAgent`](https://github.com/uiuc-kang-lab/InjecAgent) | File-browser benchmark components used by the file-browser setting. |

<a id="setup"></a>
## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Configure an OpenAI-compatible chat endpoint:

```bash
cp .env.example .env
export OPENAI_API_KEY=<your-key>
export MODEL=<model>
```

For a custom compatible endpoint:

```bash
export OPENAI_BASE_URL=<compatible-api-base-url>
```

<a id="run-experiments"></a>
## 🚀 Run Experiments

File-browser experiments:

```bash
RUN_ID=<run_id> MODEL=<model> bash file/exp/run_batch.sh
```

Web-browser WebTrap experiments:

```bash
EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=<run_id> MODEL=<model> \
  bash web/exp/run_batch_psaa_attacks.sh
```

Web-browser baseline experiments:

```bash
EXPERIMENT_ROOT=web/runs EXPERIMENT_ID=<run_id> MODEL=<model> \
  bash web/exp/run_batch_baseline_attacks.sh
```

Useful entry documents:

- [`web/exp/README.md`](./web/exp/README.md)
- [`web/psaa/README.md`](./web/psaa/README.md)
- [`web/baseline/RUNBOOK.md`](./web/baseline/RUNBOOK.md)
- [`file/exp/README.md`](./file/exp/README.md)

Focused checks:

```bash
pytest \
  utils/defenses/test_prompt_injection_baselines.py \
  web/psaa/test_inject_from_attack_case.py \
  web/baseline/test_inject_from_attack_case.py \
  web/exp/05_evaluate/test_extract_paths.py \
  web/exp/osr/test_make_osr_user_tasks.py \
  file/exp/attack/test_common.py
```

<a id="citation"></a>
## 📚 Citation

If you use this code, please cite the WebTrap paper. GitHub can also read the repository citation metadata from [`CITATION.cff`](./CITATION.cff).

```bibtex
@misc{liu2026webtrap,
  title         = {WebTrap: Stealthy Mid-Task Hijacking of Browser Agents During Navigation},
  author        = {Liu, Zhichao and Pan, Wenbo and Yu, Haining and Gao, Ge and Zhu, Tianqing and Jia, Xiaohua},
  year          = {2026},
  eprint        = {2605.08310},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2605.08310},
  url           = {https://arxiv.org/abs/2605.08310}
}
```

<a id="license"></a>
## 📄 License

This repository is released under the MIT License. Third-party projects keep their own licenses.
