# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--use_wall",
    action="store_true",
    default=False,
    help="Enable wall obstacles in the environment. Changes observation space from 42 to 45.",
)
parser.add_argument(
    "--track",
    type=str,
    default="complex",
    choices=["complex", "lemniscate"],
    help="Track layout to use: 'complex' or 'lemniscate'.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import time
import torch
import yaml

# Add third_parties directory to Python path to use local skrl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/third_parties"))

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import src.isaac_quad_sim2real.tasks # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # set use_wall configuration
    env_cfg.use_wall = args_cli.use_wall
    # Update observation spaces based on use_wall (since __post_init__ already ran)
    if env_cfg.use_wall:
        env_cfg.observation_spaces = {"ego": 45, "adversary": 45}
        env_cfg.state_space = 90
    else:
        env_cfg.observation_spaces = {"ego": 42, "adversary": 42}
        env_cfg.state_space = 84
    # set track configuration
    env_cfg.track = args_cli.track

    try:
        experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    
    # Load rewards configuration from saved training env.yaml
    # The checkpoint path structure is: .../experiment_dir/checkpoints/agent_xxx.pt
    # So we need to go up two directories to get to experiment root, then to params/env.yaml
    experiment_dir = os.path.dirname(os.path.dirname(resume_path))
    env_yaml_path = os.path.join(experiment_dir, "params", "env.yaml")
    
    if os.path.exists(env_yaml_path):
        print(f"[INFO] Loading rewards configuration from: {env_yaml_path}")
        with open(env_yaml_path, 'r') as f:
            # Use full_load to handle Python-specific tags like !!python/tuple
            saved_env_cfg = yaml.full_load(f)
            if 'rewards' in saved_env_cfg:
                env_cfg.rewards = saved_env_cfg['rewards']
                print(f"[INFO] Loaded rewards: {env_cfg.rewards}")
            else:
                print("[WARNING] No rewards found in saved env.yaml")
    else:
        print(f"[WARNING] env.yaml not found at {env_yaml_path}, using default empty rewards")

    # # pennovation circle track
    # if hasattr(env_cfg, 'viewer'):
    #     env_cfg.viewer.resolution = (1920, 1080)  # Full HD resolution
    #     env_cfg.viewer.eye = (-1.6, 11.5, 7.0)     # Camera position
    #     env_cfg.viewer.lookat = (1.0, 1.3, -1.5)   # Look at point

    # # pennovation wall track
    # if hasattr(env_cfg, 'viewer'):
    #     env_cfg.viewer.resolution = (1920, 1080)  # Full HD resolution
    #     env_cfg.viewer.eye = (-3.1, 0.4, 7.2)  # Camera position
    #     env_cfg.viewer.lookat = (0.2, 4.0, 0.8)  # Look at point

    # complex track
    if hasattr(env_cfg, 'viewer'):
        env_cfg.viewer.resolution = (1920, 1080)  # Full HD resolution
        env_cfg.viewer.eye = (11.7, -0.5, 7.5)  # Camera position
        env_cfg.viewer.lookat = (-2.9, -0.3, -1.5)  # Look at point

    # # lemniscate track
    # if hasattr(env_cfg, 'viewer'):
    #     env_cfg.viewer.resolution = (1920, 1080)  # Full HD resolution
    #     env_cfg.viewer.eye = (10.7, 5.4, 7.2)  # Camera position
    #     env_cfg.viewer.lookat = (-2.7, 5.5, -0.3)  # Look at point
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
