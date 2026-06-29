# DRL4NDN — Deep Reinforcement Learning for Adaptive NDN Forwarding

> **Paper:** *A systematic evaluation of deep reinforcement learning for adaptive NDN forwarding: A Gym-based comparative study*
> 
> **Journal:** Computer Networks
> 
> **Authors:** *Mustapha Reda Senouci, Badis Djamaa, and Yakoub Mordjana*
> 
> **Paper link:** *https://doi.org/10.1016/j.comnet.2026.112473*

---

## Overview

This repository provides the full implementation accompanying the paper. It introduces **DRL4NDN**, a custom [Gymnasium](https://gymnasium.farama.org/) environment that simulates an **NDN (Named Data Networking) router** and provides a systematic comparison of forwarding strategies across multiple agent families:

| Agent Family | Algorithms |
|---|---|
| Deep Reinforcement Learning (DRL) | PPO, A2C, DQN, TRPO, QR-DQN |
| Multi-Armed Bandit (MAB) | ε-Greedy, UCB, Thompson Sampling, Softmax |
| Contextual MAB (CMAB) | LinUCB |
| Baselines | Oracle (Best-Face), Random |

The environment models a router choosing among `N` forwarding faces per incoming Interest packet. Each face has stochastic RTT, NACK, and timeout behaviour. Seven distinct network scenarios are supported, four observation state representations, and two reward functions, enabling a rigorous combinatorial evaluation.

---

## Requirements

### System
- Linux (tested on Ubuntu 20.04+)
- Conda (Miniconda / Anaconda)

### Installation

**Step 1 — Create the conda environment (handles PyTorch and most scientific dependencies):**
```bash
conda create --name DRL -c pytorch -c conda-forge --file conda-requirements.txt
conda activate DRL
```

**Step 2 — Install remaining pip packages:**
```bash
pip install tqdm==4.66.1 \
            stable-baselines3==2.2.1 \
            sb3-contrib==2.2.1 \
            rich==13.7.0 \
            pytz==2023.3.post1 \
            pygments==2.17.2 \
            mdurl==0.1.2 \
            markdown-it-py==3.0.0 \
            fsspec==2023.12.0
```

**Key dependencies at a glance:**

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.1.1 (CPU) | DRL training backend |
| stable-baselines3 | 2.2.1 | PPO, A2C, DQN |
| sb3-contrib | 2.2.1 | TRPO, QR-DQN |
| gymnasium | via conda | RL environment interface |
| numpy | 1.26.0 | Numerical operations |
| matplotlib | via conda | Plotting |

---

## Quick Start

### 1. Verify the environment works
Run a quick sanity check using the oracle agent (no training required):
```bash
cd DRL4NDN/
python test.py
```
This instantiates the `NDNRouterEnv` with `scenario=static_with_one_optimal_face`, `seed=79`, and `num_faces=3`, then runs 1 000 steps using the oracle best-face selector. Expected output: per-step reward printout and a total reward summary.

### 2. Run full experiments (all scenarios, all seeds)

Each bash script inside `scripts/` launches the full experiment grid for one agent family. Run them from the `scripts/` directory:

```bash
cd DRL4NDN/scripts/

bash Random.sh   # Random baseline
bash Best.sh     # Oracle upper-bound
bash MAB.sh      # MAB agents
bash CMAB.sh     # Contextual MAB (LinUCB)
bash DRL.sh      # DRL agents (PPO, A2C, DQN, TRPO, QR-DQN)
```

> **Compute note:** DRL experiments are long-running, particularly for `num_faces=30` (up to 1 000 000 training timesteps). Running on a multi-core machine or in parallel is recommended. CPU-only execution is supported by default.

### 3. Process logs and generate plots

After experiments complete, logs are written under `logs/<scenario>/<reward>/<state>/<agent_family>/`.
Use the script `drl4ndn_analysis.py` in `process_log_scripts/` to aggregate results and reproduce the paper's figures (set the `filesdirectory` in  `drl4ndn_analysis.py` to the logs directory).

---

## Repository Structure

```
DRL4NDN/
│
├── gym_ndn/                        # Custom Gymnasium package
│   ├── __init__.py                 # Registers NDNRouter-v0 with gym.make()
│   └── envs/
│       ├── __init__.py
│       └── ndn_router.py           # ★ Core environment: NDNRouterEnv
│
├── agents/                         # Agent runners (one file per agent family)
│   ├── DRL_Agents.py               # PPO, A2C, DQN, TRPO, QR-DQN
│   ├── DRL_Agents_budget.py        # DRL variant with explicit training-budget sweep
│   ├── MAB_Agents.py               # ε-Greedy, UCB, Thompson Sampling, Softmax MAB
│   ├── CMAB_Agents.py              # LinUCB (contextual MAB)
│   ├── Best_Agent.py               # Oracle: always picks the best face (upper bound)
│   └── Random_Agent.py             # Random baseline
│
├── scripts/
│   ├── Bash scripts to launch full experiment grids
│   │   ├── DRL.sh
│   │   ├── MAB.sh
│   │   ├── CMAB.sh
│   │   ├── Best.sh
│   │   └── Random.sh
│   └── process_log_scripts/               # Post-processing and plotting scripts
│       ├── drl4ndn_analysis.py         
│
├── helpers.py                      # Shared utilities: training loop, evaluation, plotting
├── test.py                        # Quick oracle-agent smoke test
├── conda-requirements.txt          # Pinned conda environment spec
└── README.txt                      # Original minimal setup note
```

### Key files in detail

**`gym_ndn/envs/ndn_router.py` — `NDNRouterEnv`**
The heart of the project. Implements a `gymnasium.Env` subclass simulating a multi-face NDN router. Key constructor parameters:

| Parameter | Options | Description |
|---|---|---|
| `state` | `state_one` … `state_four` | Observation representation |
| `reward_fct` | `simple`, `streak` | Reward function |
| `scenario` | 7 options | Network dynamics scenario |
| `num_faces` | `3`, `10`, `30` | Number of forwarding faces |
| `seed` | int | Random seed for reproducibility |

*Observation states:*
- **state_one** — 6 raw counters per face (sent interests, received data/NACKs, timeouts, avg RTT, last RTT)
- **state_two** — 2 aggregate metrics per face (success rate, moving-average RTT)
- **state_three** — 2 moving-window metrics per face (moving success rate, moving avg RTT)
- **state_four** — 4 metrics per face including per-prefix moving success rate and per-prefix moving avg RTT

*Reward functions:*
- **simple** — Normalized inverse RTT on success; −1 on NACK or timeout
- **streak** — Adds a streak bonus/penalty proportional to consecutive success/failure runs

*Scenarios:*

| Scenario | Description |
|---|---|
| `static_with_one_optimal_face` | One face always has the lowest RTT |
| `dynamic_RTT` | RTTs shift periodically across faces |
| `permanent_faulty_face` | One face is permanently degraded |
| `transient_faulty_face` | Faulty face recovers after a period |
| `permanent_face_specific` | Face performance is prefix-dependent (permanent) |
| `transient_face_specific` | Face performance is prefix-dependent (transient) |
| `streak_disruption` | Environment reacts adversarially to agent behaviour patterns |

**`helpers.py`**
Shared utilities used by all agent runners: `train_agent()`, `eval_test_agent()`, `plot_actions_over_time()`, `plot_evaluations_data()`, `plot_monitor_data()`.

**`agents/DRL_Agents.py`**
Wraps Stable-Baselines3 algorithms. Accepts CLI arguments `--reward`, `--state`, `--scenario`, `--agent`, `--face`. Trains for a face-count-dependent budget over 5 seeds and logs per-seed rewards.

**`agents/Best_Agent.py`**
Oracle agent that calls `env.get_best_face()` at each step — provides the theoretical performance ceiling.

---

## Experiment Configuration

The full grid evaluated in the paper:

| Axis | Values |
|---|---|
| Scenarios | 7 |
| Observation states | state_one, state_two, state_three, state_four |
| Reward functions | simple, streak |
| Number of faces | 3, 10, 30 |
| Random seeds | 0, 1, 2, 3, 4, ... |
| DRL agents | PPO, A2C, DQN, TRPO, QR-DQN |
| MAB agents | ε-Greedy, UCB, Thompson Sampling, Softmax |
| CMAB agents | LinUCB |

---

## Reproducibility Notes

- **Seeds:** All experiments use seeds `[0, ..., 100]`. Seeds are applied to Python `random`, NumPy, and PyTorch before each run. The environment itself also accepts a seed via `NDNRouterEnv(seed=...)`.
- **Device:** All DRL training runs on CPU (`device = "cpu"`). Results in the paper were obtained on CPU to ensure cross-machine determinism.
- **Log paths:** Agent scripts write logs relative to `../logs/`. Ensure you run agents from within the `agents/` directory, or adjust paths accordingly.
- **Expected outputs:** Each agent run produces a `.log` file with per-episode rewards and a final `Average Reward ± Std` summary line. Directory structure: `logs/<scenario>/<reward>/<state>/<agent_family>/<agent_name>/<N>faces/<seed>seed/`.
- **Post-processing:** The `drl4ndn_analysis.py` scripts expect logs in the structure above. Run them only after all simulation scripts have completed.
- **Environment registration:** `gym_ndn/__init__.py` registers `NDNRouter-v0` with Gymnasium. The `DRL4NDN/` root must be on your Python path (agent scripts handle this automatically via `sys.path.insert`).

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{drl4ndn2026,
  title = {A systematic evaluation of deep reinforcement learning for adaptive NDN forwarding: A Gym-based comparative study},
  author = {Mustapha Reda Senouci and Badis Djamaa and Yakoub Mordjana},
  journal = {Computer Networks},
  volume = {286},
  pages = {112473},
  year = {2026},
  issn = {1389-1286},
  doi = {https://doi.org/10.1016/j.comnet.2026.112473},
  url = {https://www.sciencedirect.com/science/article/pii/S1389128626004858},
}
```

---

## License

*This project is licensed under the MIT License.*
