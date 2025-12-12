# Agile Flight Emerges from Multi-Agent Competition

[![Agile Flight Emerges from Multi-Agent Competition](AgileFlight_CoverImage.png)](https://youtu.be/AIUfCbEJX6E)

This repository contains the code for training and evaluating Our multi-agent quadcopter racing policy in the paper, Agile Flight Emerges from Multi-Agent Competition.
In order to train the Dense Single (DS) and Sparse Single (SS) policies, please navigate to the [AgileFlight_SingleAgent branch](https://github.com/Jirl-upenn/AgileFlight_MultiAgent/tree/AgileFlight_SingleAgent).
## Paper and Video

Paper: [PDF](JIRL_Multi_Agent_Drone_Racing_ICRA_2026.pdf)

Video: [Youtube](https://youtu.be/AIUfCbEJX6E)

```bibtex
@article{pasumarti2025agileflightemergesmultiagentcompetition,
  title={Agile Flight Emerges from Multi-Agent Competition},
  author={Pasumarti, Bianchi, Loquercio},
  year={2025}
}
```

## Setup

### Prerequisites

- GPU with CUDA support
- NVIDIA Isaac Sim v4.5.0
- NVIDIA Isaac Lab v2.1.0
- Ubuntu 20.04 / 22.04 (recommended)

### Installation

1. Clone the repository:

```bash
# It is critical that the project repo and the Isaac Lab directory are at the same level
git clone -b AgileFlight_MultiAgent https://github.com/Jirl-upenn/isaac_quad_sim2real.git
cd isaac_quad_sim2real
```

2. Create and activate your Isaac Lab v2.1.0 conda environment ([Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))

3. Install the package and dependencies:

```bash
# Install the main package
pip install -e .
```

## Training Examples

The main training script uses a modified mappo.py from the [skrl](https://skrl.readthedocs.io/) library.

```bash
# Train Our policy on the Complex Track with walls
python scripts/skrl/ma_train_race.py \
    --task Isaac-MA-Quadcopter-Race-v0 \
    --num_envs 10240 \
    --algorithm MAPPO \
    --max_iterations 10000 \
    --headless \
    --use_wall \
    --track complex
```

```bash
# Train Our policy on the Lemniscate Track
python scripts/skrl/ma_train_race.py \
    --task Isaac-MA-Quadcopter-Race-v0 \
    --num_envs 10240 \
    --algorithm MAPPO \
    --max_iterations 10000 \
    --headless \
    --track lemniscate
```

## Evaluation

To evaluate a trained policy:

```bash
# Evaluate Our policy on the Complex Track with walls
python scripts/skrl/ma_play_race.py \
    --task Isaac-MA-Quadcopter-Race-v0 \
    --num_envs 1 \
    --algorithm MAPPO \
    --track complex \
    --use_wall \
    --checkpoint path/to/checkpoint.pt \
    --video \
    --video_length 1000 \
    --headless
```