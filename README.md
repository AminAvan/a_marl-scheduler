## Overview

1) Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

2) Mava allows researchers to experiment with multi-agent reinforcement learning (MARL) at lightning speed. The single-file JAX implementations are built for rapid research iteration - hack, modify, and test new ideas fast. Mava's [state-of-the-art algorithms][sable] scale seamlessly across devices. Created for researchers, by The Research Team at [InstaDeep](https://www.instadeep.com).

----

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

To run the code, start training, and view the results of A-MARL (IPPO_SPT: [informed_ff_ippo.py](mava/systems/ppo/anakin/informed_ff_ippo.py)) and MARL (IPPO: [ff_ippo.py](mava/systems/ppo/anakin/ff_ippo.py)) in the JobShop environment, please execute the following commands:

```bash
# for running MARL (IPPO)
python mava/systems/ppo/anakin/ff_ippo.py env=job_shop 
# for running A-MARL (IPPO+SPT)
python mava/systems/ppo/anakin/informed_ff_ippo.py env=job_shop
```

In order to see the default system configs please see the `mava/configs/` directory.
A benefit of Hydra is that configs can either be set in config yaml files or overwritten from the terminal on the fly.

Additionally, Mava developers also have a [Quickstart notebook][quickstart] that can be used to quickly create and train your first multi-agent system.

<h2>Algorithms</h2>

Mava has implementations of multiple on- and off-policy multi-agent algorithms that follow the independent learners (IL), centralised training with decentralised execution (CTDE) and heterogeneous agent learning paradigms. Aside from MARL learning paradigms, we also include implementations which follow the Anakin and Sebulba architectures to enable scalable training by default. The architecture that is relevant for a given problem depends on whether the environment being used in written in JAX or not. For more information on these paradigms, please see [here][anakin_paper].

| Algorithm  | Variants       | Continuous | Discrete | Anakin | Sebulba | Paper | Docs |
|------------|----------------|------------|----------|--------|---------|-------|------|
| PPO        | [`ff_ippo.py`](mava/systems/ppo/anakin/ff_ippo.py)   | ✅         | ✅       | ✅     | ✅      | [Link](https://arxiv.org/abs/2011.09533) | [Link](mava/systems/ppo/README.md) |
|            | [`ff_mappo.py`](mava/systems/ppo/anakin/ff_mappo.py)  | ✅         | ✅       | ✅     |         | [Link](https://arxiv.org/abs/2103.01955) | [Link](mava/systems/ppo/README.md) |
|            | [`rec_ippo.py`](mava/systems/ppo/anakin/rec_ippo.py)  | ✅         | ✅       | ✅     |         | [Link](https://arxiv.org/abs/2011.09533) | [Link](mava/systems/ppo/README.md) |
|            | [`rec_mappo.py`](mava/systems/ppo/anakin/rec_mappo.py) | ✅         | ✅       | ✅     |         | [Link](https://arxiv.org/abs/2103.01955) | [Link](mava/systems/ppo/README.md) |
| Q Learning | [`rec_iql.py`](mava/systems/q_learning/anakin/rec_iql.py)   |            | ✅       | ✅     |         | [Link](https://arxiv.org/abs/1511.08779) | [Link](mava/systems/q_learning/README.md) |
|            | [`rec_qmix.py`](mava/systems/q_learning/anakin/rec_qmix.py)  |            | ✅       | ✅     |         | [Link](https://arxiv.org/abs/1803.11485) | [Link](mava/systems/q_learning/README.md) |
| SAC        | [`ff_isac.py`](mava/systems/sac/anakin/ff_isac.py)   | ✅         |          | ✅     |         | [Link](https://arxiv.org/abs/1801.01290) | [Link](mava/systems/sac/README.md) |
|            | [`ff_masac.py`](mava/systems/sac/anakin/ff_masac.py)  | ✅         |          | ✅     |         |     | [Link](mava/systems/sac/README.md) |
|            | [`ff_hasac.py`](mava/systems/sac/anakin/ff_hasac.py)  | ✅         |          | ✅     |         | [Link](https://arxiv.org/abs/2306.10715) | [Link](mava/systems/sac/README.md) |
| MAT        | [`mat.py`](mava/systems/mat/anakin/mat.py)       | ✅         | ✅       | ✅     |         | [Link](https://arxiv.org/abs/2205.14953) | [Link](mava/systems/mat/README.md) |
| Sable      | [`ff_sable.py`](mava/systems/sable/anakin/ff_sable.py)  | ✅         | ✅       | ✅     |         | [Link](https://arxiv.org/abs/2410.01706) | [Link](mava/systems/sable/README.md) |
|            | [`rec_sable.py`](mava/systems/sable/anakin/rec_sable.py) | ✅         | ✅       | ✅     |         | [Link](https://arxiv.org/abs/2410.01706) | [Link](mava/systems/sable/README.md) |
<h2>Environments</h2>

These are the environments which Mava supports _out of the box_, to add a new environment, please use the [existing wrapper implementations](mava/wrappers/) as an example. We also indicate whether the environment is implemented in JAX or not. JAX-based environments can be used with algorithms that follow the Anakin distribution architecture, while non-JAX environments can be used with algorithms following the Sebulba architecture.


| Environment                     | Action space        | JAX | Non-JAX | Paper | JAX Source | Non-JAX Source |
|---------------------------------|---------------------|-----|-------|-------|------------|----------------|
| Mulit-Robot Warehouse                 | Discrete            | ✅   | ✅     | [Link](http://arxiv.org/abs/2006.07869)  |    [Link](https://github.com/instadeepai/jumanji/tree/main/jumanji/environments/routing/robot_warehouse)   |       [Link](https://github.com/semitable/robotic-warehouse)      |
| Level-based Foraging            | Discrete            | ✅   | ✅     | [Link](https://arxiv.org/abs/2006.07169)  |    [Link](https://github.com/instadeepai/jumanji/tree/main/jumanji/environments/routing/lbf)    |       [Link](https://github.com/semitable/lb-foraging)      |
| StarCraft Multi-Agent Challenge | Discrete            | ✅   | ✅     | [Link](https://arxiv.org/abs/1902.04043)  |    [Link](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax)    |       [Link](https://github.com/uoe-agents/smaclite)      |
| Multi-Agent Brax                          | Continuous          | ✅   |       | [Link](https://arxiv.org/abs/2003.06709)  |    [Link](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mabrax)    |             |
| Matrax                          | Discrete            | ✅   |       | [Link](https://www.cs.toronto.edu/~cebly/Papers/_download_/multirl.pdf)  |    [Link](https://github.com/instadeepai/matrax)    |             |
| Multi Particle Environments            | Discrete/Continuous | ✅   |       | [Link](https://arxiv.org/abs/1706.02275)  |    [Link](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mpe)    |            |

## Performance and Speed 🚀

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

## Code Philosophy 🧘

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

## Contributing 🤝

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

## Roadmap 🛤️

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

## Citing Mava 📚

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

## Acknowledgements 🙏

Since this repository is a clone of the original Mava, please refer to the official Mava repository at https://github.com/instadeepai/Mava for the original source and credits.

[Paper]: https://arxiv.org/pdf/2107.01460.pdf
[quickstart]: https://github.com/instadeepai/Mava/blob/develop/examples/Quickstart.ipynb
[jumanji]: https://github.com/instadeepai/jumanji
[cleanrl]: https://github.com/vwxyzjn/cleanrl
[purejaxrl]: https://github.com/luchris429/purejaxrl
[jumanji_rware]: https://instadeepai.github.io/jumanji/environments/robot_warehouse/
[jumanji_lbf]: https://github.com/sash-a/jumanji/tree/feat/lbf-truncate
[epymarl]: https://github.com/uoe-agents/epymarl
[anakin_paper]: https://arxiv.org/abs/2104.06272
[rware]: https://github.com/semitable/robotic-warehouse
[jaxmarl]: https://github.com/flairox/jaxmarl
[toward_standard_eval]: https://arxiv.org/pdf/2209.10485.pdf
[marl_eval]: https://github.com/instadeepai/marl-eval
[smax]: https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax
[sable]: https://arxiv.org/pdf/2410.01706
