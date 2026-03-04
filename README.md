# Agile Multi-Agent Reinforcement Learning (A-MARL)

> Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits. Mava allows researchers to experiment with multi-agent reinforcement learning (MARL) at lightning speed. The single-file JAX implementations are built for rapid research iteration - hack, modify, and test new ideas fast. Mava's [state-of-the-art algorithms][sable] scale seamlessly across devices. Created for researchers, by The Research Team at [InstaDeep](https://www.instadeep.com).

> **Based on the paper:** [A-MARL: Agile Multi-Agent Reinforcement Learning for Soft Real-Time Task Scheduling in Edge Computing (IEEE CASCON 2025)](https://doi-org.uproxy.library.dc-uoit.ca/10.1109/CASCON66301.2025.00053)

## Overview

Modern **Soft Real-Time Applications (SRTAs)** impose heavy computational demands on embedded devices. Offloading workloads to **Edge Computing (EC)** resources is an attractive solution, but task scheduling remains challenging due to:

- Strict timing constraints
- Vast combinatorial search spaces
- Multiple conflicting optimization objectives
- Highly dynamic and unpredictable environments

Conventional heuristic and meta-heuristic approaches struggle to adapt. Single-agent Reinforcement Learning (RL) converges too slowly on medium- and large-scale problems due to enormous action spaces.

**[A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py)** solves this by enhancing Multi-Agent PPO with **entropy-guided rule-based exploration**, using the Shortest Processing Time (SPT) algorithm to steer agents toward promising regions of the action space during early training, and seamlessly transitioning to a learned policy as confidence grows.

## Key Features

- **Entropy-Guided Exploration:** Adaptively switches between *SPT-based informed exploration* and *policy-based action selection* depending on current *policy entropy*.
- **Pseudo-Action Masking:** Prunes the effective *action space* during *high-entropy* phases, focusing exploration on domain-informed (SPT) choices.
- **Tailored Actor-Critic Architecture:** Fuses a residual MLP for feature embedding with a lightweight self-attention block to capture cross-dependencies between operations and edge resources.
- **[A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) converges 61% faster** Compared to standard [MARL](mava/systems/ppo/anakin/ff_ippo.py) baseline.
- **[A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) achieves 57% better sample efficiency** Fewer environment interactions required to reach comparable solution quality.
- **[A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) reduces convergence-time by 846.3 seconds** Significant practical improvement for real-world deployment.

## Architecture

```
A-MARL
├── Environment: JobShop (discrete action space)
│   └── Each action = (Task ID, Edge-Server ID)
│
├── Agents: One RL-agent per Edge-Server
│   └── Independent policy learning in a shared environment
│
├── Actor-Critic Network
│   ├── Residual MLP  ──── embeds SRTA & edge-server features
│   ├── Self-Attention ─── captures cross-operation/resource dependencies
│   ├── Actor Head ──────── outputs action distribution
│   └── Critic Head ─────── estimates value function
│
└── Entropy-Guided Mechanism
    ├── High Entropy  →  SPT-based informed exploration + pseudo-action masking
    └── Low Entropy   →  Learned policy exploitation
```

### Entropy Threshold

The switching condition is defined as in Eq.(14) and below:

$$H_{th} = \alpha \cdot \log_e\!\left(\left|\bigcup_{\tau \in \delta \in E} \tau\right| + 1\right)$$

where $\alpha \in (0, 1)$ is the threshold fraction and the log term represents the maximum possible entropy of the action space.


## Method Details

### 1. Pseudo-Action Masking

During high-entropy phases (early training), the policy is immature and action probabilities are nearly uniform. [A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) applies **entropy-conditioned pruning**: the action space is constrained to only SPT-guided actions, dramatically reducing exploration overhead and focusing agents on domain-informed choices.

### 2. Informed Exploration

When entropy surpasses $H_{th}$, agents navigate the state-action space using SPT rather than the learned policy. Agents assign unallocated tasks with the **shortest processing time** to available edge-servers while respecting scheduling constraints.

### 3. Policy Maturation

As training progresses and entropy decreases, [A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) seamlessly transitions to **full policy-based exploitation**; combining the benefits of SPT-guided initialization and learned optimization.

## 📊 Results

| Metric | [MARL](mava/systems/ppo/anakin/ff_ippo.py) (Baseline) | **[A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py)** | Improvement |
|--------|----------------|-------------------|-------------|
| Convergence Time | — | — | **61% faster** |
| Sample Efficiency | — | — | **57% better** |
| Convergence Time Reduction | — | — | **846.3 seconds** |
| Value Loss Stability | Moderate | Low & Stable | ✅ Better |
| Hit-Ratio (Φ) | Baseline | Higher | ✅ Better |
| Makespan | Baseline | Comparable | ✅ Maintained |

[A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) consistently outperforms state-of-the-art baselines across **all evaluated metrics** on representative SRTA scheduling scenarios.

## Installation

At the moment Mava is not meant to be installed as a library, but rather to be used as a research tool. Mava developers recommend cloning the Mava repo and installing dependencies using [uv](https://github.com/astral-sh/uv) as follows:

```bash
# Clone the repository
git clone https://github.com/instadeepai/Mava.git
cd Mava
# Create a virtual environment and install all dependencies
uv sync
# Activate the virtual environment
source .venv/bin/activate
```

Alternatively with pip, create a virtual environment and then:
```bash
pip install -e .
```

Mava developers have tested `Mava` on Python 3.11 and 3.12, but earlier versions may also work. Specifically, they use Python 3.10 for the Quickstart notebook on Google Colab since Colab uses Python 3.10 by default. Note that because the installation of JAX differs depending on your hardware accelerator,
we advise users to explicitly install the correct JAX version (see the [official installation guide](https://github.com/google/jax#installation)). For more in-depth installation guides including Docker builds and virtual environments, please see their [detailed installation guide](docs/DETAILED_INSTALL.md).

## Getting Started with the Paper's Code: Running A-MARL and MARL Methods Mentioned in the Paper

To run the code, start training, and view the results of [A-MARL](mava/systems/ppo/anakin/informed_ff_ippo.py) and [MARL](mava/systems/ppo/anakin/ff_ippo.py) in the JobShop environment, please execute the following commands:

```bash
# for running MARL (IPPO)
python mava/systems/ppo/anakin/ff_ippo.py env=job_shop 
# for running A-MARL (IPPO+SPT)
python mava/systems/ppo/anakin/informed_ff_ippo.py env=job_shop
```

In order to see the default system configs please see the `mava/configs/` directory.
A benefit of Hydra is that configs can either be set in config yaml files or overwritten from the terminal on the fly.

Additionally, Mava developers also have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first multi-agent system.

## 📄 Citation

If you find this work useful in your research, please cite:

```bibtex
@INPROCEEDINGS{11344342,
  author={Avan, Amin and Azim, Akramul and Mahmoud, Qusay H.},
  booktitle={2025 IEEE International Conference on Collaborative Advances in Software and COmputiNg (CASCON)}, 
  title={A-MARL: Agile Multi-Agent Reinforcement Learning for Soft Real-Time Task Scheduling in Edge Computing}, 
  year={2025},
  volume={},
  number={},
  pages={275-284},
  keywords={Schedules;Processor scheduling;Reinforcement learning;Dynamic scheduling;Search problems;Real-time systems;Software;Timing;Space exploration;Edge computing;Reinforcement Learning;Multi-Agent Reinforcement Learning;Edge Computing;Task Scheduling;Real-Time Systems;Real-Time Applications},
  doi={10.1109/CASCON66301.2025.00053}}

```

## Acknowledgements

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.