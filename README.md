# PPO-Based Routing for Containerized Multi-Model Serving Systems

# Abstract

Large language model (LLM) enabled software engineering services increasingly rely on heterogeneous model backends that differ in latency, cost, and output quality. Selecting a backend per request becomes a multi-objective operational decision, since the system has to weigh responsiveness and cost against response quality under serving conditions that keep shifting. This paper proposes an online routing approach for containerized multi-model serving based on Proximal Policy Optimization (PPO). The router picks one backend per request from lightweight request features and recent windowed backend performance statistics. To support real-time learning without an external evaluator, we use a deterministic structural quality proxy and combine it with latency and normalized cost penalties in a multi-component reward. We evaluate the method against several online routing baselines under identical deployment, observability, and workload conditions in a fully containerized testbed. Across a mixed workload typical of software engineering assistants, the PPO router favors higher structural output quality while keeping traffic spread across backends. It improves mean structural quality by roughly 32–65% over the bandit baselines under balanced reward settings, at the cost of accepting higher latency.

The code is available at: https://github.com/ResponsibleAILab/PPO_Based_Router

## RL-Router Architecture

The diagram below shows the end-to-end routing path. Incoming prompts are tagged by a planner and sent to a selected LLM backend; the observed latency, cost, and quality signals feed the online PPO policy updates.

<p align="left">
  <img src="Images/Polished_RL Router Architecture_v3.4.drawio.svg" alt="PPO Based Router" width="60%">
  <br><em>PPO Based Router Architecture</em>
</p>

## Key Metrics

| Metric          | Meaning                                                                                                                |
| --------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Structural Quality (Q)** | Deterministic, task-specific structural-completeness score (code syntax markers, `assert`/`test_` for tests, word count for explanations). Higher is better. |
| **Mean Latency**| Wall-to-wall time (seconds) from request receipt to backend response, including container and GPU scheduling overhead. Lower is better. |
| **Mean Reward** | Scalar reward `R = αQ − βL − γC` received per request under the active coefficient configuration.                       |
| **Mean Cost**   | Weighted-token cost proxy for the chosen backend (generated tokens scaled by per-backend weights; only the weights are normalized). Lower is cheaper. |
| **Backend Usage** | How often each backend is selected, which shows whether a method spreads traffic or collapses onto one backend.       |

## Key Results

**Structural quality.**
On code tasks, where backend specialization matters most, PPO reaches the highest mean structural quality (0.637) by a wide margin over the bandit baselines, including MOUCB (0.385), which uses the same multi-objective reward. The gap is statistically significant against all five baselines (Mann-Whitney U, one-sided, p < 0.001).

**Backend diversity.**
Every bandit method, MOUCB included, converges onto a single dominant backend (90–99% of requests) within each task category (the one exception is ε-greedy on explain at 58%). PPO is the only method that keeps real traffic on all three backends, distributing requests rather than collapsing onto one — on code it spreads across all three with Mistral as the plurality choice (~56%), and Mistral is likewise its plurality pick on explain and tests. This comes out of the online reward feedback alone.

**Latency trade-off.**
PPO runs at higher latency than the latency-minimizing bandit baselines on every task (ε-greedy on explain is the exception, where cold model-load overhead pushes its mean to 53.68s). It gives up some raw reward and responsiveness in exchange for better structural output quality.

**Takeaway.**
Multi-model routing is a multi-objective problem. Latency- and cost-sensitive deployments may be better off with a bandit policy, while applications that care most about response quality benefit from RL-based routing that can use backend heterogeneity and request context. The fact that MOUCB shares PPO's reward and still collapses onto one backend points to PPO's advantage coming from conditioning on multi-dimensional state, not from the reward design.

### Per-Category Results (Baseline reward: α=1.0, β=1.0, γ=0.001)

Lower latency/cost and higher reward/quality are better. Backend 0 = Mistral-7B-Instruct, Backend 1 = LLaMA-3-8B-Instruct, Backend 2 = CodeLLaMA-7B-Instruct.

#### Code

| Method        |   n |  Mean Lat. (s) | p95 (s) | Mean Rew. | Mean Qual. | Mean Cost |
| :------------ | --: | -------------: | ------: | --------: | ---------: | --------: |
| ε-greedy      | 300 |          11.73 |   23.29 |  −11.341  |     0.433  |     40.72 |
| Softmax       | 300 |          10.29 |   19.63 |   −9.923  |     0.410  |     39.17 |
| UCB           | 300 |           9.98 |   19.23 |   −9.610  |     0.405  |     37.91 |
| Thompson      | 301 |          10.93 |   21.73 |  −10.514  |     0.484  |     68.88 |
| MOUCB         | 301 |           9.43 |   19.01 |   −9.085  |     0.385  |     37.33 |
| **Ours (PPO)**| 300 |          16.30 |   25.05 |  −15.720  |   **0.637**|     52.44 |

#### Explain

| Method        |   n |  Mean Lat. (s) | p95 (s) | Mean Rew. | Mean Qual. | Mean Cost |
| :------------ | --: | -------------: | ------: | --------: | ---------: | --------: |
| ε-greedy      | 138 |          53.68 |  247.20 |  −53.215  |     0.519  |     55.05 |
| Softmax       | 101 |          16.05 |   21.40 |  −15.393  |     0.709  |     51.04 |
| UCB           | 101 |          13.37 |   22.12 |  −12.789  |     0.643  |     64.34 |
| Thompson      | 101 |          13.48 |   22.30 |  −12.897  |     0.653  |     65.06 |
| MOUCB         | 100 |          14.57 |   22.10 |  −14.015  |     0.653  |     94.94 |
| **Ours (PPO)**| 100 |          16.48 |   25.27 |  −15.903  |     0.640  |     65.34 |

#### Tests

| Method        |   n |  Mean Lat. (s) | p95 (s) | Mean Rew. | Mean Qual. | Mean Cost |
| :------------ | --: | -------------: | ------: | --------: | ---------: | --------: |
| ε-greedy      | 108 |          19.32 |   25.97 |  −18.659  |     0.709  |     51.45 |
| Softmax       | 108 |          17.83 |   22.88 |  −17.193  |     0.684  |     46.79 |
| UCB           | 108 |          17.55 |   22.47 |  −16.917  |     0.685  |     46.94 |
| Thompson      | 108 |          18.04 |   23.34 |  −17.369  |     0.779  |    109.25 |
| MOUCB         | 107 |          16.66 |   21.29 |  −16.018  |     0.706  |     64.46 |
| **Ours (PPO)**| 107 |          19.79 |   26.17 |  −19.152  |     0.702  |     68.37 |

Note: small differences in `n` come from occasional request failures, which are dropped before aggregation.

## Method Overview

Routing is set up as a request-level Markov Decision Process. At each request *t* the router reads a state `s_t` (per-backend sliding-window latency stats — mean, p50, p95, and a saturation proxy — concatenated with prompt length and a one-hot task tag), picks one backend `a_t ∈ {0, …, K−1}`, and gets a scalar reward once the request finishes:

```
R_t = α · Q_t − β · L_t − γ · C_t
```

Here `Q_t` is the deterministic task-specific quality score, `L_t` is end-to-end latency, and `C_t` is the normalized token cost. PPO trains an actor-critic policy on-policy from a rolling experience buffer using the clipped surrogate objective. The five baselines (ε-greedy, Softmax, Thompson Sampling, UCB, and MOUCB) run over the same observation space, action space, and reward, so any difference in behavior comes from the algorithm rather than from what information it had access to.

## Backends

| Backend | Model                  | Role                                   | Cost Weight |
| :------ | :--------------------- | :------------------------------------- | ----------: |
| 0       | Mistral-7B-Instruct    | Smaller, lower-latency model           | 0.5         |
| 1       | LLaMA-3-8B-Instruct    | General-purpose, strong overall quality| 1.0         |
| 2       | CodeLLaMA-7B-Instruct  | Code-specialized model                 | 0.7         |

The three backends are served by a single shared Ollama instance, each fronted by its own MCP adapter service (`mcp_a`, `mcp_b`, `mcp_c`) that targets Ollama with a different `MODEL_NAME`. All services are brought up with Docker Compose on a shared bridge network. A `planner` service tags each prompt (`code`, `explain`, or `tests`) and forwards it to the `rl_router`, which picks one backend (adapter) per request. Prometheus, Grafana, cAdvisor, and DCGM handle monitoring.

| Service      | Port   | Role                                              |
| :----------- | :----- | :------------------------------------------------ |
| `planner`    | `9000` | Tags prompts and drives `/plan_run`               |
| `rl_router`  | `8080` | Routing policy + `/infer`, `/health`, `/policy/*` |
| `prometheus` | `9090` | Metrics scraping                                  |
| `grafana`    | `3000` | Dashboards                                         |

The active policy is chosen by layering a per-method override file over the base `docker-compose.yml` (for example, `docker-compose.ppo.yml` sets `POLICY: "ppo"`).

## System Requirements

Experiments are run from WSL2 on a Windows host with Docker Desktop and a set of Bash scripts. They also run on native Linux.

- Docker Desktop with WSL2 integration (or Docker Engine + the NVIDIA Container Toolkit on Linux)
- An NVIDIA GPU is recommended for LLM serving; 32 GB RAM; tens of GB free disk for images, models, and logs
- Inside WSL: Bash, Python 3.9+, `curl`, `jq`, and the Docker CLI

Check your environment:
```bash
docker ps
python3 --version
```

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Make the experiment scripts executable (one time):
```bash
chmod +x scripts/run_all.sh
chmod +x scripts/run_eval_ppo.sh
chmod +x scripts/run_eval_epsilon.sh
chmod +x scripts/run_eval_softmax.sh
chmod +x scripts/run_eval_ucb.sh
chmod +x scripts/run_eval_thompson.sh
chmod +x scripts/train.sh
```

## To Run

### 1. Train the PPO policy

PPO trains online by driving the planner with composite prompts and checkpointing to `checkpoints/ppo.pt`:

```bash
# bring the stack up in PPO mode first
docker compose -f docker-compose.yml -f docker-compose.ppo.yml up -d --build

# train (ROUNDS defaults to 500, checkpoint every 100 prompts)
./scripts/train.sh 500 100
```

### 2. Evaluate the bandit baselines

`run_all.sh` evaluates softmax, ucb, and thompson across all three datasets, resetting the stack with `docker compose down -v` between runs:

```bash
./scripts/run_all.sh
```

> Each run is capped at 300 requests by default. Override it with `MAX_REQUESTS=500 ./scripts/run_all.sh`.

### 3. Evaluate PPO and ε-greedy

These two have their own scripts, run once per dataset and tag:

```bash
./scripts/run_eval_ppo.sh     datasets/code_prompts.txt    code
./scripts/run_eval_ppo.sh     datasets/explain_prompts.txt explain
./scripts/run_eval_ppo.sh     datasets/tests_prompts.txt   tests

./scripts/run_eval_epsilon.sh datasets/code_prompts.txt    code
./scripts/run_eval_epsilon.sh datasets/explain_prompts.txt explain
./scripts/run_eval_epsilon.sh datasets/tests_prompts.txt   tests
```

### 4. Evaluate MOUCB

MOUCB is the multi-objective bandit that shares PPO's reward. There's a helper that runs all three datasets and then analyzes them:

```bash
./scripts/run_eval_moucb_all.sh
```

For a single dataset and tag: `./scripts/run_eval_moucb.sh datasets/code_prompts.txt code`.

### Reward-sensitivity sweep (ablation)

To reproduce the reward ablation, sweep the PPO reward coefficients over the preset configurations (paper default, quality-up, latency-up, balanced):

```bash
./scripts/run_reward_sweep_ppo.sh all
```

### Running a single method manually

```bash
docker compose down -v || true
docker compose -f docker-compose.yml -f docker-compose.softmax.yml up -d --build
./scripts/run_eval_softmax.sh datasets/code_prompts.txt code
```

Swap `softmax` for `ucb` or `thompson` as needed. Each eval script copies its live log to a timestamped file, for example `logs/route_log_ppo_code_<timestamp>.jsonl`.

## Logs and Outputs

Per-request traces are written to `logs/` as JSONL:
```
logs/route_log_<method>.jsonl                       # live log, cleared at the start of each run
logs/route_log_<method>_<tag>_<timestamp>.jsonl     # timestamped snapshot per dataset
```

## Analyzing Results

`analyze_results.py` picks up all timestamped logs in `--logdir` (matching `route_*_{epsilon,ppo,softmax,ucb,thompson,moucb}_*.jsonl`) and writes the CSV summaries, comparison tables, and figures used in the paper:

```bash
python3 scripts/analyze_results.py --logdir logs --outdir results
```

This writes a `summaries_<timestamp>.csv`, `comparison_table.md` / `comparison_table.tex`, and PDF figures for overall backend usage, per-request Pareto fronts, reward and latency distributions, and per-tag latency/reward and backend usage into `results/`. Run `python3 scripts/analyze_results.py -h` for the options.

For the significance testing in the paper (paired bootstrap CIs, p-values, and Cohen's *d* of PPO against each baseline per tag):

```bash
python3 scripts/analyze_stats.py --logdir logs --outdir stats_results
```

## Reproducibility Notes

- Reference hardware: NVIDIA RTX 4060 GPU (8 GB VRAM), Intel i5-14400 CPU, 32 GB RAM, Ubuntu 22.04 LTS, Docker Engine 24.0.
- Backends run on a single `ollama` service using quantized GGUF weights, fronted by the `mcp_a/b/c` adapters. Because 8 GB of VRAM cannot hold all three 7–8B models at once, Ollama keeps one model resident and loads the selected backend on demand, evicting the previous one. Requests are processed sequentially, so only the active model runs at any instant.
- Model load time on a backend switch is included in the reported end-to-end latency (this is why switch-heavy runs, e.g. ε-greedy on explain, show much heavier latency tails).
- Routing logic and logging run on the host CPU; the GPU is reserved for backend inference.
- All backend containers get the same CPU, memory, and GPU limits, and persistent volumes are cleared between experiments for a clean start.
- A reward-sensitivity ablation (α=2.5, β=0.5, γ=0.002) is included to check how robust PPO's quality advantage is to the coefficient choice.

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@inproceedings{SteeleDingFeng2026PPORouter,
  author    = {Michael Steele and Junhua Ding and Yunhe Feng},
  title     = {{PPO}-Based Routing for Containerized Multi-Model Serving Systems},
  booktitle = {Proceedings of the 26th IEEE International Conference on Software Quality, Reliability, and Security (QRS)},
  series    = {Lecture Notes in Computer Science},
  year      = {2026},
  publisher = {Springer}
}
```
