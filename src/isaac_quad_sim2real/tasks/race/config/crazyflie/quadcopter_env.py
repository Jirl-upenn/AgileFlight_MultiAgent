# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from matplotlib import pyplot as plt
from collections import deque
import math

from pxr import Gf, UsdGeom, Sdf, UsdPhysics, PhysxSchema
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat

from typing import List
from dataclasses import dataclass, field
from datetime import datetime
import csv

from scipy.spatial.transform import Rotation as R

##
# Drone config
##
# from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
CRAZYFLIE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    collision_group=0,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"usd/cf2x.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

D2R = np.pi / 180.0
R2D = 180.0 / np.pi
PLOT_UPDATE_FREQ = 5

class DelayBuffer:
    def __init__(self, delay_time_s, policy_rate_hz, num_envs, device=None):
        """
        Per-environment delay buffer with fixed delay and full batch operations.

        Args:
            delay_time_s (float): Target delay time (same for all environments)
            policy_rate_hz (float): Frequency at which the policy is called (Hz)
            num_envs (int): Number of parallel environments
            device (str or torch.device): Device to allocate tensors on
        """
        self.delay = max(0, math.ceil(delay_time_s * policy_rate_hz))
        self.num_envs = num_envs
        self.action_dim = 4
        self.device = device

        self.queue = torch.zeros((self.delay, num_envs, self.action_dim),
                                 dtype=torch.float32, device=device)

        self.valid = torch.zeros((self.delay, num_envs), dtype=torch.bool, device=device)
        self.idx = torch.zeros(num_envs, dtype=torch.long, device=device)

        default = torch.tensor([-1.0, 0.0, 0.0, 0.0], device=device)
        self.default_action = default.unsqueeze(0).expand(num_envs, -1).clone()

    def save_and_retrieve(self, cmd):
        """
        Enqueue current commands and return delayed outputs.

        Args:
            cmd (torch.Tensor): shape (num_envs, 4)

        Returns:
            torch.Tensor: delayed commands of shape (num_envs, 4)
        """
        if self.delay == 0:
            return cmd

        env_ids = torch.arange(self.num_envs, device=self.device)

        delayed_cmd = torch.where(
            self.valid[self.idx, env_ids].unsqueeze(1),
            self.queue[self.idx, env_ids],
            self.default_action
        )

        self.queue[self.idx, env_ids] = cmd.detach()
        self.valid[self.idx, env_ids] = True
        self.idx = (self.idx + 1) % self.delay

        # delayed_cmd[:, :3] = cmd[:, :3]
        return delayed_cmd

    def reset(self, env_ids):
        """
        Reset the delay buffer for selected environments.

        Args:
            env_ids (Iterable[int]): List of environment indices to reset
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.valid[:, env_ids] = False
        self.idx[env_ids] = 0
        self.queue[:, env_ids] = self.default_action[env_ids].unsqueeze(0)

class ResetBuffer:
    def __init__(self, max_size, motor_speed_min, motor_speed_max, device='cuda', noise_std=None):
        self.max_size = max_size
        self.motor_speed_min = motor_speed_min
        self.motor_speed_max = motor_speed_max
        self.device = device
        self.index = 0
        self.full = False
        self.noise_std = noise_std or {}

        self.buffer = torch.zeros((max_size, 17), dtype=torch.float32, device=device)
        self.wp_idx_buffer = torch.zeros((max_size, 1), dtype=torch.int, device=device)

    def append_batch(self, batch_tensor, wp_idx_tensor):
        end = self.index + batch_tensor.shape[0]

        if end <= self.max_size:
            self.buffer[self.index:end] = batch_tensor
            self.wp_idx_buffer[self.index:end] = wp_idx_tensor
        else:
            first_part = self.max_size - self.index
            self.buffer[self.index:] = batch_tensor[:first_part]
            self.buffer[:end % self.max_size] = batch_tensor[first_part:]
            self.wp_idx_buffer[self.index:] = wp_idx_tensor[:first_part]
            self.wp_idx_buffer[:end % self.max_size] = wp_idx_tensor[first_part:]

        self.index = end % self.max_size
        if end >= self.max_size:
            self.full = True

    def get_all(self):
        if not self.full:
            return self.buffer[:self.index], self.wp_idx_buffer[:self.index]
        else:
            return (
                torch.cat((self.buffer[self.index:], self.buffer[:self.index]), dim=0),
                torch.cat((self.wp_idx_buffer[self.index:], self.wp_idx_buffer[:self.index]), dim=0)
            )

    def sample(self, batch_size):
        buffer_len = len(self)

        if batch_size > buffer_len:
            idx = torch.randint(0, buffer_len, (batch_size,), device=self.device)
        else:
            idx = torch.randperm(buffer_len, device=self.device)[:batch_size]

        data_all, wp_all = self.get_all()
        batch = data_all[idx].clone()
        wp_idx = wp_all[idx].clone()

        if self.noise_std:
            batch[:, 0:3] += torch.randn_like(batch[:, 0:3]) * self.noise_std.get('pos', 0.0)
            batch[:, 3:7] = self._apply_quaternion_noise(batch[:, 3:7], self.noise_std.get('quat', 0.0))
            batch[:, 7:10] += torch.randn_like(batch[:, 7:10]) * self.noise_std.get('lin_vel', 0.0)
            batch[:, 10:13] += torch.randn_like(batch[:, 10:13]) * self.noise_std.get('ang_vel', 0.0)
            batch[:, 13:17] = self._apply_motor_speeds_noise(batch[:, 13:17], self.noise_std.get('motor_speeds', 0.0))

        return {
            'pos': batch[:, 0:3],
            'quat': batch[:, 3:7],
            'lin_vel': batch[:, 7:10],
            'ang_vel': batch[:, 10:13],
            'motor_speeds': batch[:, 13:17],
            'curr_wp_idx':  wp_idx.squeeze(-1)
        }

    def _apply_quaternion_noise(self, q, std_rad):
        batch_size = q.shape[0]
        axis = torch.randn((batch_size, 3), device=q.device)
        axis = axis / torch.norm(axis, dim=-1, keepdim=True)
        angle = torch.randn((batch_size, 1), device=q.device) * std_rad
        sin = torch.sin(angle / 2)
        cos = torch.cos(angle / 2)
        q_noise = torch.cat((cos, axis * sin), dim=-1)  # (w, x, y, z)
        return self._quaternion_multiply(q, q_noise)

    def _apply_motor_speeds_noise(self, motor_speeds, std):
        noise = torch.randn_like(motor_speeds) * std
        noisy_speeds = motor_speeds + noise
        return torch.clamp(noisy_speeds, min=self.motor_speed_min, max=self.motor_speed_max)

    def _quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack((
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ), dim=-1)

    def __len__(self):
        return self.max_size if self.full else self.index

GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

class Curriculum:
    def __init__(self, start: int, end: int, step_interval: int, initial_value: float, final_value: float):
        self.start = start
        self.end = end
        self.step_interval = step_interval
        self.initial_value = initial_value
        self.final_value = final_value

    def get_value(self, iteration: int) -> float:
        if iteration <= self.start:
            return self.initial_value
        elif iteration >= self.end:
            return self.final_value
        else:
            total_steps = (self.end - self.start) // self.step_interval
            step_count = (iteration - self.start) // self.step_interval
            alpha = step_count / total_steps
            value = self.initial_value + alpha * (self.final_value - self.initial_value)
            return value

class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.
        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

@dataclass
class GateModelCfg:
    usd_path: str = "./usd/gate.usda"
    prim_name: str = "gate"
    gate_side: float = 1.0
    scale = [1.0, gate_side, gate_side]

@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    use_wall = True
    track_name = 'lemniscate'

    # env
    episode_length_s = 30.0             # episode_length = episode_length_s / dt / decimation
    action_space = 4
    observation_space = (
        (3 if use_wall else 0) +   # global position (only with wall)
         3 +                        # linear velocity
         9 +                        # attitude matrix
        12 +                        # relative desired position vertices waypoint 1
        12                          # relative desired position vertices waypoint 2
    )
    state_space = 0
    debug_vis = True

    sim_rate_hz = 500
    policy_rate_hz = 50
    pid_loop_rate_hz = 500
    decimation = sim_rate_hz // policy_rate_hz
    pid_loop_decimation = sim_rate_hz // pid_loop_rate_hz

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)
    gate_model: GateModelCfg = field(default_factory=GateModelCfg)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        force_threshold=0.0,
        # filter_prim_paths_expr=["/World/envs/env_.*/Wall"],
    )

    beta = 1.0         # 1.0 for no smoothing, 0.0 for no update

    # Reset variables
    min_altitude = 0.1
    max_altitude = 3.0
    max_time_on_ground = 1.5

    # motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    # PID parameters
    kp_omega_rp = 250.0
    ki_omega_rp = 500.0
    kd_omega_rp = 2.5
    i_limit_rp = 33.3

    kp_omega_y = 120.0
    ki_omega_y = 16.70
    kd_omega_y = 0.0
    i_limit_y = 166.7

    body_rate_scale_xy = 100.0 * D2R
    body_rate_scale_z = 200.0 * D2R

    # Parameters from train.py or play.py
    is_train = None

    motor_noise_cv_start = 1500
    motor_noise_cv_end = 5000
    motor_noise_cv_step_interval = 100
    motor_noise_cv_initial_value = 0.0
    motor_noise_cv_final_value = 0.0

    k_aero_xy = 9.1785e-7
    k_aero_z = 10.311e-7

    max_tilt_thresh = 150 * D2R

    delay_time_s = 0.0

    max_n_laps = 3

    rewards = {}

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        self._all_target_models_paths: List[List[str]] = []
        self._models_paths_initialized: bool = False
        self.target_models_prim_base_name: str | None = None

        super().__init__(cfg, render_mode, **kwargs)

        self.iteration = 0
        self.motor_noise_std = 0.0

        if len(cfg.rewards) > 0:
            self.rew = cfg.rewards
        elif self.cfg.is_train:
            raise ValueError("rewards not provided")

        # Initialize tensors
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._undelayed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_yaw = torch.zeros(self.num_envs, device=self.device)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._wrench_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._motor_speeds_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_omega_meas = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_omega_err = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._last_distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._yaw_n_laps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._idx_wp = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._n_gates_passed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        self._crashed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # Motor dynamics
        self.cfg.thrust_to_weight = 3.15
        r = self.cfg.arm_length * np.sqrt(2.0) / 2.0
        self._rotor_positions = torch.tensor(
            [
                [ r,  r, 0],
                [ r, -r, 0],
                [-r, -r, 0],
                [-r,  r, 0]
            ],
            dtype=torch.float32,
            device=self.device
        )
        self._rotor_directions = torch.tensor([1, -1, 1, -1], device=self.device)
        self.k = self.cfg.k_m / self.cfg.k_eta

        self.f_to_TM = torch.cat(
            [
                torch.tensor([[1, 1, 1, 1]], device=self.device),
                torch.cat(
                    [
                        torch.linalg.cross(self._rotor_positions[i], torch.tensor([0.0, 0.0, 1.0], device=self.device)).view(-1, 1)[0:2] for i in range(4)
                    ],
                    dim=1,
                ).to(self.device),
                self.k * self._rotor_directions.view(1, -1),
            ],
            dim=0
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

        if not self.cfg.is_train:
            self.save_csv = True
            if self.save_csv:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.file_path = f'logs/csv/log_{timestamp}.csv'
                with open(self.file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'time',
                        # Observations
                        'lin_vel_x', 'lin_vel_y', 'lin_vel_z',
                        'attitude_00', 'attitude_01', 'attitude_02',
                        'attitude_10', 'attitude_11', 'attitude_12',
                        'attitude_20', 'attitude_21', 'attitude_22',
                        'curr_gate_v1_x', 'curr_gate_v1_y', 'curr_gate_v1_z',
                        'curr_gate_v2_x', 'curr_gate_v2_y', 'curr_gate_v2_z',
                        'curr_gate_v3_x', 'curr_gate_v3_y', 'curr_gate_v3_z',
                        'curr_gate_v4_x', 'curr_gate_v4_y', 'curr_gate_v4_z',
                        'next_gate_v1_x', 'next_gate_v1_y', 'next_gate_v1_z',
                        'next_gate_v2_x', 'next_gate_v2_y', 'next_gate_v2_z',
                        'next_gate_v3_x', 'next_gate_v3_y', 'next_gate_v3_z',
                        'next_gate_v4_x', 'next_gate_v4_y', 'next_gate_v4_z',
                        # Actions
                        'action_thrust', 'action_bodyrate_x', 'action_bodyrate_y', 'action_bodyrate_z',
                        # Extra
                        'global_x', 'global_y', 'global_z',
                        'ang_vel_x', 'ang_vel_y', 'ang_vel_z',
                        'seed',
                        'n_gates',
                        'gates_passed',
                        'n_crashes'
                    ])

            self.draw_plots = False
            if self.draw_plots:
                self.data_fig, self.data_axes = plt.subplots(6, 1, figsize=(10, 10))
                self.data_fig.show()

                self.max_len_deque = 100
                self.rp_history = deque(maxlen=self.max_len_deque)
                self.yaw_history = deque(maxlen=self.max_len_deque)
                self.z_history = deque(maxlen=self.max_len_deque)
                self.v_history = deque(maxlen=self.max_len_deque)
                self.twr_history = deque(maxlen=self.max_len_deque)
                self.actions_history = deque(maxlen=self.max_len_deque)
                self.undelayed_actions_history = deque(maxlen=self.max_len_deque)
                self.n_steps = 0
                self.rp_lines = [self.data_axes[0].plot([], [], label=f"{legend}")[0] for legend in ["Roll", "Pitch"]]
                self.yaw_line, = self.data_axes[1].plot([], [], 'g', label="Yaw")
                self.z_line, = self.data_axes[2].plot([], [], 'y', label="Z")
                self.v_line, = self.data_axes[3].plot([], [], 'r', label="v")
                self.twr_lines = [self.data_axes[4].plot([], [], label=f"{legend}")[0] for legend in ["Desired TWR", "Commanded TWR"]]
                self.actions_lines = [self.data_axes[5].plot([], [], label=f"{legend}")[0] for legend in ["Thrust", "Roll rate", "Pitch rate", "Yaw rate"]]
                self.undelayed_actions_lines = [self.data_axes[5].plot([], [], linestyle='--', color=self.actions_lines[i].get_color())[0] for i in range(4)]

                self.body_rates_fig, self.body_rates_axes = plt.subplots(3, 1, figsize=(10, 10))
                self.body_rates_fig.show()
                self.roll_rate_history = deque(maxlen=self.max_len_deque)
                self.pitch_rate_history = deque(maxlen=self.max_len_deque)
                self.yaw_rate_history = deque(maxlen=self.max_len_deque)
                self.body_rates_r_line = [self.body_rates_axes[0].plot([], [], label=f"{legend}")[0] for legend in ["Desired Roll rate", "Commanded Roll rate"]]
                self.body_rates_p_line = [self.body_rates_axes[1].plot([], [], label=f"{legend}")[0] for legend in ["Desired Pitch rate", "Commanded Pitch rate"]]
                self.body_rates_y_line = [self.body_rates_axes[2].plot([], [], label=f"{legend}")[0] for legend in ["Desired Yaw rate", "Commanded Yaw rate"]]

                if self.num_envs > 1:
                    self.draw_plots = False
                    plt.close(self.data_fig)
                    plt.close(self.body_rates_fig)

                # Configure subplots
                for ax, title in zip(self.data_axes, ["RP History", "Yaw History", "Z History", "Velocity", "TWRs", "Actions History"]):
                    # ax.set_title(title)
                    ax.set_xlabel("Time Step")
                    if title == "RP History":
                        ax.set_ylabel("Angle (°)")
                    if title == "Yaw History":
                        ax.set_ylabel("Angle (°)")
                    elif title == "Actions History":
                        ax.set_ylabel("Action")
                    elif title == "Z History":
                        ax.set_ylabel("z (m)")
                    elif title == "Velocity":
                        ax.set_ylabel("v (m/s)")
                    elif title == "Commanded TWR":
                        ax.set_ylabel("TWR")
                    ax.legend(loc="upper left")
                    ax.grid(True)

                for ax, title in zip(self.body_rates_axes, ["Roll rate History", "Pitch rate History", "Yaw rate History"]):
                    # ax.set_title(title)
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Angular velocity (°/s)")
                    ax.legend(loc="upper left")
                    ax.grid(True)

                plt.tight_layout()
                plt.ion()  # interactive mode

        # Logging
        if self.cfg.is_train:
            keys = [key.split("_reward_scale")[0] for key in self.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in keys
            }

        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)

        self.set_debug_vis(self.cfg.debug_vis)

        self._motor_noise_std_cv = Curriculum(
            start=self.cfg.motor_noise_cv_start,
            end=self.cfg.motor_noise_cv_end,
            step_interval=self.cfg.motor_noise_cv_step_interval,
            initial_value=self.cfg.motor_noise_cv_initial_value,
            final_value=self.cfg.motor_noise_cv_final_value,
        )

        self._pose_drone_wrt_gate = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)

        noise_std = {'pos': 0.05, 'quat': 0.1, 'lin_vel': 0.05, 'ang_vel': 0.1, 'motor_speeds': 100.0}
        self._reset_buffer = ResetBuffer(100_000, self.cfg.motor_speed_min, self.cfg.motor_speed_max, device=self.device, noise_std=noise_std)

        self._num_updates_buffer = 1000
        self._num_updates_buffer_ground = 100
        self._start_buffer_population = 200000
        self._start_dynamic_reset = 300000
        self._initial_wp = 0

        self._n_run = 0

        self._K_aero = torch.zeros(self.num_envs, 3, device=self.device)

        self._kp_omega = torch.zeros(self.num_envs, 3, device=self.device)
        self._ki_omega = torch.zeros(self.num_envs, 3, device=self.device)
        self._kd_omega = torch.zeros(self.num_envs, 3, device=self.device)

        self._tau_m = torch.zeros(self.num_envs, 4, device=self.device)

        self._omega_err_integral = torch.zeros(self.num_envs, 3, device=self.device)

        self._thrust_to_weight = torch.zeros(self.num_envs, device=self.device)

        self.delay_commands = DelayBuffer(self.cfg.delay_time_s, self.cfg.policy_rate_hz, self.num_envs, self.device)

        # Values for randomization
        if self.cfg.is_train:
            # TWR
            self._twr_min = self.cfg.thrust_to_weight
            self._twr_max = self.cfg.thrust_to_weight

            # Aerodynamics
            self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
            self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
            self._k_aero_z_min = self.cfg.k_aero_z * 0.5
            self._k_aero_z_max = self.cfg.k_aero_z * 2.0

            # PID gains
            self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
            self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
            self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
            self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
            self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.7
            self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.2

            self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
            self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
            self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
            self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
            self._kd_omega_y_min = self.cfg.kd_omega_y * 0.7
            self._kd_omega_y_max = self.cfg.kd_omega_y * 1.2

            # Motor parameters
            self._tau_m_min = self.cfg.tau_m * 0.2
            self._tau_m_max = self.cfg.tau_m * 2.0
        else:
            self._twr_min = self._twr_max = self.cfg.thrust_to_weight

            self._k_aero_xy_min = self._k_aero_xy_max = self.cfg.k_aero_xy
            self._k_aero_z_min = self._k_aero_z_max = self.cfg.k_aero_z

            self._kp_omega_rp_min = self._kp_omega_rp_max = self.cfg.kp_omega_rp
            self._ki_omega_rp_min = self._ki_omega_rp_max = self.cfg.ki_omega_rp
            self._kd_omega_rp_min = self._kd_omega_rp_max = self.cfg.kd_omega_rp
            self._kp_omega_y_min = self._kp_omega_y_max = self.cfg.kp_omega_y
            self._ki_omega_y_min = self._ki_omega_y_max = self.cfg.ki_omega_y
            self._kd_omega_y_min = self._kd_omega_y_max = self.cfg.kd_omega_y

            self._tau_m_min = self._tau_m_max = self.cfg.tau_m

    def update_iteration(self, iter):
        self.iteration = iter

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = GOAL_MARKER_CFG.copy()
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self._gate_model_cfg_data = getattr(self.cfg, 'gate_model', {})
        model_usd_file_path = self._gate_model_cfg_data.usd_path

        self._target_models_prim_base_name = self._gate_model_cfg_data.prim_name

        model_scale = Gf.Vec3f(*self._gate_model_cfg_data.scale)

        stage = get_current_stage()
        env0_root_path_str = "/World/envs/env_0"

        d = self._gate_model_cfg_data.gate_side / 2
        self._local_square = torch.tensor([
            [0,  d,  d],
            [0, -d,  d],
            [0, -d, -d],
            [0,  d, -d]
        ], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1, 1)

        #########################

        tracks = {
            'complex': [
                [ 1.5,  3.5, 0.75, 0.0, 0.0, -0.7854],
                [-1.5,  3.5, 0.75, 0.0, 0.0,  0.7854],
                [-2.0, -3.5, 2.00, 0.0, 0.0,  1.5708],
                [-2.0, -3.5, 0.75, 0.0, 0.0, -1.5708],
                [ 1.0, -1.0, 2.00, 0.0, 0.0,  3.1415],
                [ 1.0, -3.5, 0.75, 0.0, 0.0,  0.0000],
            ],
            'lemniscate': [
                [ 1.5, 3.50, 0.75, 0.0, 0.0, -1.57],
                [ 0.0, 5.25, 1.50, 0.0, 0.0,  0.00],
                [-2.0, 7.00, 0.75, 0.0, 0.0, -1.57],
                [ 1.5, 7.00, 0.75, 0.0, 0.0,  1.57],
                [ 0.0, 5.25, 1.50, 0.0, 0.0,  0.00],
                [-2.0, 3.50, 0.75, 0.0, 0.0,  1.57],
            ]
        }

        self._waypoints = torch.tensor(tracks[self.cfg.track_name], device=self.device)

        self._normal_vectors = torch.zeros(self._waypoints.shape[0], 3, device=self.device)
        self._waypoints_quat = torch.zeros(self._waypoints.shape[0], 4, device=self.device)

        if self.cfg.use_wall:
            walls = {
                'lemniscate': [                      # lemniscate
                    [ 0.0, 7.65,  0.00, 1.3, 0.2, 3.0],      # x, y, alpha, length, width, height
                    [-1.0,  6.0, -0.78, 3.0, 0.2, 3.0],
                ],
                'complex': [
                    [-2.00,  2.00, 1.57, 2.0, 0.1, 8.0],
                    [ 0.25, -2.50, 1.57, 1.5, 0.1, 8.0],
                    [ 3.00, -0.25, 1.57, 4.0, 0.1, 8.0],
                    [ 0.00,  1.00, 1.57, 2.0, 0.1, 8.0]
                ]
            }

            for i, wall in enumerate(walls[self.cfg.track_name]):
                xc, yc, alpha, length, width, height = wall

                wall_object = RigidObject(
                    RigidObjectCfg(
                        prim_path=f"/World/envs/env_.*/Wall_{i}",
                        spawn=sim_utils.CuboidCfg(
                            size=(width, length, height),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                                kinematic_enabled=True
                            ),
                            collision_props=sim_utils.CollisionPropertiesCfg(),    # uncomment this to enable collisions
                            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.2),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(xc, yc, height / 2.0),
                            rot=(np.cos(alpha / 2.0), 0.0, 0.0, np.sin(alpha / 2.0))),
                    )
                )
                self.scene.rigid_objects[f"wall_{i}"] = wall_object

        if not stage.GetPrimAtPath(env0_root_path_str):
            UsdGeom.Xform.Define(stage, Sdf.Path(env0_root_path_str))

        for i, waypoint_data in enumerate(self._waypoints):
            position_tensor = waypoint_data[0:3]
            euler_angles_tensor = waypoint_data[3:6]

            euler_np = euler_angles_tensor.cpu().numpy()
            rot_from_euler = R.from_euler('xyz', euler_np)

            # scipy version (1.6.1) does not accept 'scalar_first'
            quat_xyzw = rot_from_euler.as_quat()  # shape: (4,) as [x, y, z, w]
            quat_wxyz = np.roll(quat_xyzw, shift=1)  # now [w, x, y, z]
            self._waypoints_quat[i, :] = torch.tensor(quat_wxyz, device=self.device, dtype=torch.float32)

            # self._waypoints_quat[i, :] = torch.tensor(rot_from_euler.as_quat(scalar_first=True), device=self.device, dtype=torch.float32)
            rotmat_np_gate = rot_from_euler.as_matrix()
            gate_normal_np = rotmat_np_gate[:, 0] 
            self._normal_vectors[i, :] = torch.tensor(gate_normal_np, device=self.device, dtype=torch.float32)
            current_gate_normal_world = Gf.Vec3d(float(gate_normal_np[0]), float(gate_normal_np[1]), float(gate_normal_np[2])).GetNormalized()

            current_gate_pose_position = Gf.Vec3d(
                float(position_tensor[0]), float(position_tensor[1]), float(position_tensor[2])
            )
            quat_numpy_array_gate = euler_angles_to_quat(euler_angles_tensor.cpu().numpy())
            current_gate_pose_orientation_gd = Gf.Quatd(
                float(quat_numpy_array_gate[0]), float(quat_numpy_array_gate[1]),
                float(quat_numpy_array_gate[2]), float(quat_numpy_array_gate[3])
            )

            model_pose_xform_name = f"{self._target_models_prim_base_name}_{i}"
            model_pose_xform_path = f"{env0_root_path_str}/{model_pose_xform_name}"
            scaled_ref_xform_name = "scaled_model_ref"
            scaled_ref_xform_path = f"{model_pose_xform_path}/{scaled_ref_xform_name}"

            # 1. Create external Xform for the model pose
            usd_geom_pose_xform_obj = UsdGeom.Xform.Define(stage, Sdf.Path(model_pose_xform_path))
            model_pose_xform_prim = usd_geom_pose_xform_obj.GetPrim()

            if not model_pose_xform_prim or not model_pose_xform_prim.IsValid():
                continue

            xformable_pose_gate = UsdGeom.Xformable(model_pose_xform_prim)
            xformable_pose_gate.ClearXformOpOrder()
            op_orient_pose_gate = xformable_pose_gate.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            op_orient_pose_gate.Set(current_gate_pose_orientation_gd)
            op_translate_pose_gate = xformable_pose_gate.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            op_translate_pose_gate.Set(current_gate_pose_position)
            xformable_pose_gate.SetXformOpOrder([op_translate_pose_gate, op_orient_pose_gate])

            # 2. Create Xform for the scaled reference model
            usd_geom_scaled_ref_xform_obj = UsdGeom.Xform.Define(stage, Sdf.Path(scaled_ref_xform_path))
            model_scaled_ref_xform_prim = usd_geom_scaled_ref_xform_obj.GetPrim()
            if not model_scaled_ref_xform_prim or not model_scaled_ref_xform_prim.IsValid():
                continue

            xformable_scaled_ref_gate = UsdGeom.Xformable(model_scaled_ref_xform_prim)
            xformable_scaled_ref_gate.ClearXformOpOrder()
            op_scale_model_gate = xformable_scaled_ref_gate.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            op_scale_model_gate.Set(model_scale)
            xformable_scaled_ref_gate.SetXformOpOrder([op_scale_model_gate])

            # 3. Create gates
            references_api_gate = model_scaled_ref_xform_prim.GetReferences()
            references_api_gate.AddReference(assetPath=model_usd_file_path)

            # 4. Apply collisions to gates
            for child_prim in model_scaled_ref_xform_prim.GetChildren():
                for mesh_prim in child_prim.GetChildren():
                    if mesh_prim.GetTypeName() == "Mesh":
                        # Apply rigid body to the parent xform
                        rb_api = UsdPhysics.RigidBodyAPI.Apply(child_prim)
                        rb_api.CreateKinematicEnabledAttr().Set(True)

                        # apply collision API to the mesh
                        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
                        collision_api.CreateCollisionEnabledAttr().Set(True)

                        # apply mesh collision API with convex decomposition
                        # this creates multiple convex shapes that preserve the gate opening
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
                        mesh_collision_api.CreateApproximationAttr().Set("convexDecomposition")

            arrow_length = 0.5
            arrow_body_radius = 0.01
            arrow_head_radius = 0.03
            arrow_head_height_factor = 0.25
            arrow_color_gf = Gf.Vec3f(0.0, 1.0, 0.0)

            arrow_xform_name = f"{model_pose_xform_name}_normal_arrow"
            arrow_xform_path = f"{env0_root_path_str}/{arrow_xform_name}"

            arrow_parent_xform_geom = UsdGeom.Xform.Define(stage, Sdf.Path(arrow_xform_path))
            arrow_parent_xform_prim = arrow_parent_xform_geom.GetPrim()

            if arrow_parent_xform_prim and arrow_parent_xform_prim.IsValid():
                default_arrow_up_axis = Gf.Vec3d(0.0, 1.0, 0.0)
                inverted_gate_normal_world = -current_gate_normal_world 

                arrow_rotation = Gf.Rotation(default_arrow_up_axis, inverted_gate_normal_world)
                arrow_orientation_quat = arrow_rotation.GetQuat()

                xformable_arrow_parent = UsdGeom.Xformable(arrow_parent_xform_prim)
                xformable_arrow_parent.ClearXformOpOrder()

                op_orient_arrow = xformable_arrow_parent.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                op_orient_arrow.Set(arrow_orientation_quat)

                op_translate_arrow = xformable_arrow_parent.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                op_translate_arrow.Set(current_gate_pose_position)

                xformable_arrow_parent.SetXformOpOrder([op_translate_arrow, op_orient_arrow])

                body_height = arrow_length * (1.0 - arrow_head_height_factor)
                head_height = arrow_length * arrow_head_height_factor

                arrow_body_path = f"{arrow_xform_path}/body"
                body_geom_usd = UsdGeom.Cylinder.Define(stage, Sdf.Path(arrow_body_path))
                body_geom_usd.GetAxisAttr().Set(UsdGeom.Tokens.y)
                body_geom_usd.GetRadiusAttr().Set(arrow_body_radius)
                body_geom_usd.GetHeightAttr().Set(body_height)
                UsdGeom.XformCommonAPI(body_geom_usd.GetPrim()).SetTranslate(Gf.Vec3d(0.0, body_height / 2.0, 0.0))

                arrow_head_path = f"{arrow_xform_path}/head"
                head_geom_usd = UsdGeom.Cone.Define(stage, Sdf.Path(arrow_head_path))
                head_geom_usd.GetAxisAttr().Set(UsdGeom.Tokens.y)
                head_geom_usd.GetRadiusAttr().Set(arrow_head_radius)
                head_geom_usd.GetHeightAttr().Set(head_height)
                UsdGeom.XformCommonAPI(head_geom_usd.GetPrim()).SetTranslate(Gf.Vec3d(0.0, body_height + head_height / 2.0, 0.0))

                body_primvars_api = UsdGeom.PrimvarsAPI(body_geom_usd.GetPrim())
                body_primvars_api.CreatePrimvar("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set([arrow_color_gf])

                head_primvars_api = UsdGeom.PrimvarsAPI(head_geom_usd.GetPrim())
                head_primvars_api.CreatePrimvar("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.constant).Set([arrow_color_gf])

    def _compute_motor_speeds(self, wrench_des):
        f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        motor_speed_squared = f_des / self.cfg.k_eta
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

        return motor_speeds_des

    def _get_moment_from_ctbr(self, actions):
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, 0] = self.cfg.body_rate_scale_xy * actions[:, 1]  # roll_rate
        omega_des[:, 1] = self.cfg.body_rate_scale_xy * actions[:, 2]  # pitch_rate
        omega_des[:, 2] = self.cfg.body_rate_scale_z  * actions[:, 3]  # yaw_rate

        omega_meas = self._robot.data.root_ang_vel_b

        omega_err = omega_des - omega_meas

        self._omega_err_integral += omega_err / self.cfg.pid_loop_rate_hz
        if self.cfg.i_limit_rp > 0 or self.cfg.i_limit_y > 0:
            limits = torch.tensor(
                [self.cfg.i_limit_rp, self.cfg.i_limit_rp, self.cfg.i_limit_y],
                device=self._omega_err_integral.device
            )
            self._omega_err_integral = torch.clamp(
                self._omega_err_integral,
                min=-limits,
                max=limits
            )

        omega_int = self._omega_err_integral

        self._previous_omega_meas = torch.where(
            torch.abs(self._previous_omega_meas) < 0.0001,
            omega_meas,
            self._previous_omega_meas
        )
        omega_meas_dot = (omega_meas - self._previous_omega_meas) * self.cfg.pid_loop_rate_hz
        self._previous_omega_meas = omega_meas.clone()

        omega_dot = (
            self._kp_omega * omega_err +
            self._ki_omega * omega_int -
            self._kd_omega * omega_meas_dot
        )

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    ##########################################################
    ### Functions called in direct_rl_env.py in this order ###
    ##########################################################

    def _pre_physics_step(self, actions: torch.Tensor):
        self._undelayed_actions = actions.clone().clamp(-1.0, 1.0)    # actions come directly from the NN
        self._undelayed_actions = self.cfg.beta * self._undelayed_actions + (1 - self.cfg.beta) * self._previous_actions

        # add delay
        self._actions = self.delay_commands.save_and_retrieve(self._undelayed_actions)

        self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * self._robot_weight * self._thrust_to_weight
        self.pid_loop_counter = 0

    def _apply_action(self):
        if self.pid_loop_counter % self.cfg.pid_loop_decimation == 0:
            self._wrench_des[:, 1:] = self._get_moment_from_ctbr(self._actions)
            self._motor_speeds_des = self._compute_motor_speeds(self._wrench_des)
        self.pid_loop_counter += 1

        motor_accel = (self._motor_speeds_des - self._motor_speeds) / self._tau_m
        self._motor_speeds += motor_accel * self.physics_dt

        # add noise to motor speeds
        if not self.cfg.is_train:
            motor_noise_std = self.cfg.max_motor_noise_std
        else:
            motor_noise_std = self._motor_noise_std_cv.get_value(self.iteration)
        self._motor_speeds += torch.randn_like(self._motor_speeds) * motor_noise_std

        self._motor_speeds = self._motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max) # Motor saturation
        motor_forces = self.cfg.k_eta * self._motor_speeds ** 2
        wrench = torch.matmul(motor_forces, self.f_to_TM.t())

        # Compute drag
        lin_vel_b = self._robot.data.root_com_lin_vel_b
        theta_dot = torch.sum(self._motor_speeds, dim=1, keepdim=True)
        drag = -theta_dot * self._K_aero.unsqueeze(0) * lin_vel_b# * self._gravity_magnitude

        self._thrust[:, 0, :] = drag
        self._thrust[:, 0, 2] += wrench[:, 0]
        self._moment[:, 0, :] = wrench[:, 1:]

        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        drone_pose = self._robot.data.root_link_state_w[:, :3]
        self._pose_drone_wrt_gate, _ = subtract_frame_transforms(self._waypoints[self._idx_wp, :3] + self._terrain.env_origins,
                                                                 self._waypoints_quat[self._idx_wp, :],
                                                                 drone_pose)

        cond_gate_inside = (torch.abs(self._pose_drone_wrt_gate[:, 1]) < self._gate_model_cfg_data.gate_side / 2 - 0.1) & \
                           (torch.abs(self._pose_drone_wrt_gate[:, 2]) < self._gate_model_cfg_data.gate_side / 2 - 0.1)

        cond_gate_through = (self._pose_drone_wrt_gate[:, 0] < 0.0) & (self._prev_x_drone_wrt_gate > 0.0)
        cond_gate = cond_gate_through & ~cond_gate_inside

        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        cond_h_min_time = torch.logical_and(
            self._robot.data.root_link_pos_w[:, 2] < self.cfg.min_altitude, \
            episode_time > self.cfg.max_time_on_ground
        )

        cond_max_h = self._robot.data.root_link_pos_w[:, 2] > self.cfg.max_altitude

        cond_crashed = self._crashed > 100

        # -------------------------
        # Condition: drone stuck
        # -------------------------
        # threshold on displacement (meters)
        stuck_threshold = 0.05
        # how many steps allowed without moving
        max_stuck_steps = 400

        # initialize buffer if not existing
        if not hasattr(self, "_stuck_counter"):
            self._stuck_counter = torch.zeros(drone_pose.shape[0], dtype=torch.long, device=drone_pose.device)
            self._prev_pos = drone_pose.clone()

        # displacement norm wrt previous step
        displacement = torch.norm(drone_pose - self._prev_pos, dim=1)

        # update counter: increment if below threshold, reset if moved enough
        self._stuck_counter = torch.where(
            displacement < stuck_threshold,
            self._stuck_counter + 1,
            torch.zeros_like(self._stuck_counter),
        )

        # save current position for next call
        self._prev_pos = drone_pose.clone()

        cond_stuck = self._stuck_counter > max_stuck_steps
        # -------------------------

        died = (
            cond_max_h
          | cond_h_min_time
          | cond_crashed
          | cond_stuck
        #   | cond_gate
        )

        # timeout conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if not self.cfg.is_train:
            time_out = time_out | ((self._n_gates_passed - 1) // (self._waypoints.shape[0]) >= self.cfg.max_n_laps)

        if not self.cfg.is_train and (died or time_out):
            print(f'cond_max_h = {cond_max_h.item()}')
            print(f'cond_h_min_time = {cond_h_min_time.item()}')
            print(f'cond_crashed = {cond_crashed.item()}')
            print(f'timeout = {time_out.item()}')
            print(f'episode length = {self.episode_length_buf.item()}')
            print()

        # populate reset buffer
        if self.iteration > self._start_buffer_population:
            valid_envs = ~(died | time_out | self._crashed) & (episode_time > 3.0)

            if valid_envs.sum() > self._num_updates_buffer:
                selected_idx = torch.nonzero(valid_envs, as_tuple=False).squeeze(-1)
                sampled_idx = selected_idx[torch.randperm(len(selected_idx), device=self.device)[:self._num_updates_buffer]]
                valid_envs = torch.zeros_like(valid_envs, dtype=torch.bool)
                valid_envs[sampled_idx] = True

            # 1) build the data tensors with drone positions
            obs_tensor = torch.cat([
                self._robot.data.root_link_pos_w[valid_envs] - self._terrain.env_origins[valid_envs, :3],  # pos (3)
                self._robot.data.root_quat_w[valid_envs],                                                  # quat (4)
                self._robot.data.root_com_lin_vel_b[valid_envs],                                           # lin_vel (3)
                self._robot.data.root_com_ang_vel_b[valid_envs],                                           # ang_vel (3)
                self._motor_speeds[valid_envs]                                                             # motor_speeds (4)
            ], dim=-1)  # total shape: (B1, 17)
            wp_idx_tensor = self._idx_wp[valid_envs].unsqueeze(-1).to(torch.int)  # shape: (B, 1)

            self._reset_buffer.append_batch(obs_tensor, wp_idx_tensor)

            # 2) build the data tensors with initial ground positions
            default_root_state = self._robot.data.default_root_state[0].repeat(self._num_updates_buffer_ground, 1)

            x_local = torch.empty(self._num_updates_buffer_ground, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(self._num_updates_buffer_ground, device=self.device).uniform_(-1.0, 1.0)

            x0 = self._waypoints[self._initial_wp, 0]
            y0 = self._waypoints[self._initial_wp, 1]
            theta = self._waypoints[self._initial_wp, -1]
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            default_root_state[:, 0] = x0 - x_rot
            default_root_state[:, 1] = y0 - y_rot
            default_root_state[:, 2] = 0.0
            obs_tensor = torch.cat([
                default_root_state,
                torch.zeros(default_root_state.shape[0], 4, device=default_root_state.device)
            ], dim=-1)  # total shape: (B2, 17)
            wp_idx_tensor = self._initial_wp * torch.ones((self._num_updates_buffer_ground, 1), dtype=torch.int, device=self._idx_wp.device)

            self._reset_buffer.append_batch(obs_tensor, wp_idx_tensor)

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        # check to change setpoint
        dist_to_gate = torch.linalg.norm(self._pose_drone_wrt_gate, dim=1)
        gate_passed = (dist_to_gate < 1.0) & \
                      (self._pose_drone_wrt_gate[:, 0] < 0.0) & \
                      (self._prev_x_drone_wrt_gate > 0.0) & \
                      (torch.abs(self._pose_drone_wrt_gate[:, 1]) < self._gate_model_cfg_data.gate_side / 2) & \
                      (torch.abs(self._pose_drone_wrt_gate[:, 2]) < self._gate_model_cfg_data.gate_side / 2)
        self._prev_x_drone_wrt_gate = self._pose_drone_wrt_gate[:, 0].clone()
        ids_gate_passed = torch.where(gate_passed)[0]

        self._n_gates_passed[ids_gate_passed] += 1

        self._idx_wp[ids_gate_passed] = (self._idx_wp[ids_gate_passed] + 1) % self._waypoints.shape[0]

        lap_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        lap_completed[ids_gate_passed] = (self._n_gates_passed[ids_gate_passed] > self._waypoints.shape[0]) & \
                                        ((self._n_gates_passed[ids_gate_passed] % self._waypoints.shape[0]) == 1)

        self._desired_pos_w[ids_gate_passed, :2] = self._waypoints[self._idx_wp[ids_gate_passed], :2]
        self._desired_pos_w[ids_gate_passed, :2] += self._terrain.env_origins[ids_gate_passed, :2]
        self._desired_pos_w[ids_gate_passed, 2] = self._waypoints[self._idx_wp[ids_gate_passed], 2]

        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        self._last_distance_to_goal[ids_gate_passed] = 1.05 * distance_to_goal[ids_gate_passed].clone()

        drone_pos = self._robot.data.root_link_pos_w

        contact_forces = self._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.episode_length_buf > 100).int()
        self._crashed = self._crashed + crashed * mask

        if self.cfg.is_train:
            progress = self._last_distance_to_goal - distance_to_goal
            self._last_distance_to_goal = distance_to_goal.clone()

            roll_pitch_rates = torch.sum(torch.square(self._actions[:, 1:3]), dim=1)
            yaw_rate = torch.square(self._actions[:, 3])

            yaw_des = torch.atan2(self._desired_pos_w[:, 1] - drone_pos[:, 1], self._desired_pos_w[:, 0] - drone_pos[:, 0])
            delta_cam = (self.unwrapped_yaw - yaw_des + torch.pi) % (2 * torch.pi) - torch.pi
            perception = torch.exp(-4 * delta_cam**4)

            attitude_mat = matrix_from_quat(self._robot.data.root_quat_w)
            cos_tilt = attitude_mat[:, 2, 2]
            tilt_angle = torch.acos(cos_tilt)
            tilt_arg = self.rew["max_tilt_reward_scale"] * (torch.exp(1 * (tilt_angle - self.cfg.max_tilt_thresh) / self.cfg.max_tilt_thresh) - 1.0)
            tilt = -torch.where(tilt_arg > 0, tilt_arg, 0)

            rewards = {
                "gate_passed": gate_passed * self.rew['gate_passed_reward_scale'],
                "progress_goal": progress * self.rew['progress_goal_reward_scale'],
                "roll_pitch_rates": roll_pitch_rates * self.rew['roll_pitch_rates_reward_scale'] * self.step_dt,
                "yaw_rate": yaw_rate * self.rew["yaw_rate_reward_scale"] * self.step_dt,
                "lap_completed": lap_completed * 100.0 * self.rew['lap_completed_reward_scale'],

                "perception": perception * self.rew['perception_reward_scale'] * self.step_dt,

                "crash": crashed * self.rew['crash_reward_scale'],

                "max_tilt": tilt * self.step_dt,
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.reset_terminated, torch.ones_like(reward) * self.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if self.cfg.is_train:
            # Logging
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.extras["log"] = dict()
            self.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
            extras["Metrics/motor_noise_std"] = self._motor_noise_std_cv.get_value(self.iteration)
            self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        ######################
        if not self._models_paths_initialized:
            num_models_per_env = self._waypoints.size(0)
            model_prim_names_in_env = [f"{self.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self._all_target_models_paths = []
            for env_path in self.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self._all_target_models_paths.append(paths_for_this_env)

            self._models_paths_initialized = True
        ######################

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_yaw[env_ids] = 0.0
        self._motor_speeds[env_ids] = 0.0
        self._previous_omega_meas[env_ids] = 0.0
        self._previous_omega_err[env_ids] = 0.0
        self._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self._robot.data.default_joint_pos[env_ids]     # not important
        joint_vel = self._robot.data.default_joint_vel[env_ids]     #
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self._robot.data.default_root_state[env_ids]   # [pos, quat, lin_vel, ang_vel] in local environment frame. Shape is (num_instances, 13)

        # reset buffer
        if self.iteration < self._start_dynamic_reset:
            waypoint_indices = torch.randint(0, self._waypoints.shape[0], (n_reset,), device=self.device, dtype=self._idx_wp.dtype)

            # Get random starting poses behind waypoints
            x0_wp = self._waypoints[waypoint_indices][:, 0]
            y0_wp = self._waypoints[waypoint_indices][:, 1]
            theta = self._waypoints[waypoint_indices][:, -1]

            z_wp = self._waypoints[waypoint_indices][:, 2]

            x_local = torch.empty(n_reset, device=self.device).uniform_(-2.0, -0.5)
            y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)
            z_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)

            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            initial_x = x0_wp - x_rot
            initial_y = y0_wp - y_rot
            initial_z = z_local + z_wp

            # Reset robots state
            default_root_state[:, 0] = initial_x
            default_root_state[:, 1] = initial_y
            default_root_state[:, 2] = initial_z
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

            initial_yaw = torch.atan2(-initial_y, -initial_x)
            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                initial_yaw + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
            )
            default_root_state[:, 3:7] = quat

            if self.iteration > 800:
                percent_ground = 0.1

                ground_mask = torch.rand(n_reset, device=self.device) < percent_ground
                ground_local_ids = torch.nonzero(ground_mask, as_tuple=False).squeeze(-1)

                if ground_local_ids.numel() > 0:
                    x_local = torch.empty(len(ground_local_ids), device=self.device).uniform_(-3.0, -0.5)
                    y_local = torch.empty(len(ground_local_ids), device=self.device).uniform_(-1.0, 1.0)

                    x0_wp = self._waypoints[self._initial_wp, 0]
                    y0_wp = self._waypoints[self._initial_wp, 1]
                    theta = self._waypoints[self._initial_wp, -1]

                    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                    x_rot = cos_theta * x_local - sin_theta * y_local
                    y_rot = sin_theta * x_local + cos_theta * y_local

                    x0 = x0_wp - x_rot
                    y0 = y0_wp - y_rot
                    z0 = torch.full((len(ground_local_ids),), 0.05, device=self.device)

                    yaw0 = torch.atan2(-y0, -x0) + torch.empty(len(ground_local_ids), device=self.device).uniform_(-0.15, 0.15)
                    quat0 = quat_from_euler_xyz(
                        torch.zeros(len(ground_local_ids), device=self.device),
                        torch.zeros(len(ground_local_ids), device=self.device),
                        yaw0,
                    )

                    default_root_state[ground_local_ids, 0] = x0 + self._terrain.env_origins[env_ids[ground_local_ids], 0]
                    default_root_state[ground_local_ids, 1] = y0 + self._terrain.env_origins[env_ids[ground_local_ids], 1]
                    default_root_state[ground_local_ids, 2] = z0 + self._terrain.env_origins[env_ids[ground_local_ids], 2]
                    default_root_state[ground_local_ids, 3:7] = quat0

                    waypoint_indices[ground_local_ids] = self._initial_wp

        else:
            # Sample starting poses from ResetBuffer
            data = self._reset_buffer.sample(n_reset)

            waypoint_indices = data['curr_wp_idx']

            default_root_state[:, 0:3] = data['pos'] + self._terrain.env_origins[env_ids]
            default_root_state[:, 3:7] = data['quat']
            default_root_state[:, 7:10] = data['lin_vel']
            default_root_state[:, 10:13] = data['ang_vel']

            self._motor_speeds[env_ids] = data['motor_speeds']

        if not self.cfg.is_train:       # TODO
            # Initial position during play
            x0 = None
            y0 = None
            z0 = None
            yaw0 = None

            if x0 == None:
                x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

                x0_wp = self._waypoints[self._initial_wp, 0]
                y0_wp = self._waypoints[self._initial_wp, 1]
                theta = self._waypoints[self._initial_wp, -1]

                cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
                x_rot = cos_theta * x_local - sin_theta * y_local
                y_rot = sin_theta * x_local + cos_theta * y_local
                x0 = x0_wp - x_rot
                y0 = y0_wp - y_rot
                z0 = 0.05
                yaw0 = torch.atan2(-y0, -x0) + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
            else:
                x0 = torch.tensor(x0, device=self.device)
                y0 = torch.tensor(y0, device=self.device)
                z0 = torch.tensor(z0, device=self.device)
                yaw0 = torch.tensor(yaw0, device=self.device)

            default_root_state = self._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            self._n_run += 1
            print(f'Run #{self._n_run}: {x0.item()}, {y0.item()}, {yaw0.item()}')

            waypoint_indices = self._initial_wp

        self._idx_wp[env_ids] = waypoint_indices

        self._desired_pos_w[env_ids, :2] = self._waypoints[waypoint_indices, :2].clone()
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = self._waypoints[waypoint_indices, 2].clone()

        self._last_distance_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids, :2] - self._robot.data.root_link_pos_w[env_ids, :2], dim=1)
        self._n_gates_passed[env_ids] = 0

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self._yaw_n_laps[env_ids] = 0

        self._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(self._waypoints[self._idx_wp[env_ids], :3] + self._terrain.env_origins[env_ids, :3],
                                                                          self._waypoints_quat[self._idx_wp[env_ids], :],
                                                                          self._robot.data.root_link_state_w[env_ids, :3])

        self._K_aero[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(self._k_aero_xy_min, self._k_aero_xy_max)
        self._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(self._k_aero_z_min, self._k_aero_z_max)

        kp_omega_rp = torch.empty(n_reset, device=self.device).uniform_(self._kp_omega_rp_min, self._kp_omega_rp_max)
        ki_omega_rp = torch.empty(n_reset, device=self.device).uniform_(self._ki_omega_rp_min, self._ki_omega_rp_max)
        kd_omega_rp = torch.empty(n_reset, device=self.device).uniform_(self._kd_omega_rp_min, self._kd_omega_rp_max)

        kp_omega_y = torch.empty(n_reset, device=self.device).uniform_(self._kp_omega_y_min, self._kp_omega_y_max)
        ki_omega_y = torch.empty(n_reset, device=self.device).uniform_(self._ki_omega_y_min, self._ki_omega_y_max)
        kd_omega_y = torch.empty(n_reset, device=self.device).uniform_(self._kd_omega_y_min, self._kd_omega_y_max)

        self._kp_omega[env_ids] = torch.stack([kp_omega_rp, kp_omega_rp, kp_omega_y], dim=1)
        self._ki_omega[env_ids] = torch.stack([ki_omega_rp, ki_omega_rp, ki_omega_y], dim=1)
        self._kd_omega[env_ids] = torch.stack([kd_omega_rp, kd_omega_rp, kd_omega_y], dim=1)

        tau_m = torch.empty(n_reset, device=self.device).uniform_(self._tau_m_min, self._tau_m_max)
        self._tau_m[env_ids] = tau_m.unsqueeze(1).repeat(1, 4)

        self._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(self._twr_min, self._twr_max)

        self.delay_commands.reset(env_ids)

        self._prev_x_drone_wrt_gate = torch.ones(self.num_envs, device=self.device)

        self._crashed[env_ids] = 0

    def _get_observations(self) -> dict:
        curr_idx = self._idx_wp % self._waypoints.shape[0]
        next_idx = (self._idx_wp + 1) % self._waypoints.shape[0]

        wp_curr_pos = self._waypoints[curr_idx, :3]                # [N, 3]
        wp_next_pos = self._waypoints[next_idx, :3]                # [N, 3]
        quat_curr = self._waypoints_quat[curr_idx]                 # [N, 4]
        quat_next = self._waypoints_quat[next_idx]                 # [N, 4]

        rot_curr = matrix_from_quat(quat_curr)  # [N, 3, 3]
        rot_next = matrix_from_quat(quat_next)  # [N, 3, 3]

        verts_curr = torch.bmm(self._local_square, rot_curr.transpose(1, 2)) + wp_curr_pos.unsqueeze(1) + self._terrain.env_origins.unsqueeze(1)
        verts_next = torch.bmm(self._local_square, rot_next.transpose(1, 2)) + wp_next_pos.unsqueeze(1) + self._terrain.env_origins.unsqueeze(1)

        waypoint_pos_b_curr, _ = subtract_frame_transforms(self._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
                                                           self._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
                                                           verts_curr.view(-1, 3))
        waypoint_pos_b_next, _ = subtract_frame_transforms(self._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
                                                           self._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
                                                           verts_next.view(-1, 3))

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self._robot.data.root_quat_w   # w, x, y, z
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                *( [self._robot.data.root_link_pos_w - self._terrain.env_origins[:, :3],] if self.cfg.use_wall else [] ),     # global position
                self._robot.data.root_com_lin_vel_b,                                      # linear velocity
                attitude_mat.view(attitude_mat.shape[0], -1),                             # attitude matrix
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),               # relative desired position point 1
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),               # relative desired position point 2
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self._previous_yaw
        self._previous_yaw = yaw_w
        self._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.unwrapped_yaw = yaw_w + 2 * np.pi * self._yaw_n_laps

        self._previous_actions = self._actions.clone()

        if not self.cfg.is_train and self.save_csv:
            with open(self.file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [
                    round((self.episode_length_buf[0] * self.cfg.sim.dt * self.cfg.decimation).item(), 4),
                    *[round(x, 4) for x in self._robot.data.root_com_lin_vel_b[0].tolist()],
                    *[round(x, 4) for x in attitude_mat.view(-1).tolist()],
                    *[round(x, 4) for x in waypoint_pos_b_curr.view(-1).tolist()],
                    *[round(x, 4) for x in waypoint_pos_b_next.view(-1).tolist()],
                    *[round(x, 4) for x in self._actions.view(-1).tolist()],
                    *[round(x, 4) for x in self._robot.data.root_link_state_w[0, :3].tolist()],
                    *[round(x, 4) for x in self._robot.data.root_ang_vel_b[0].tolist()],
                    self.cfg.seed,
                    self._waypoints.shape[0],
                    self._n_gates_passed[0].item(),
                    self._crashed.item()
                ]
                writer.writerow(row)

        if not self.cfg.is_train and self.draw_plots:
            self.n_steps += 1

            roll_w = wrap_to_pi(rpy[0])
            pitch_w = wrap_to_pi(rpy[1])
            z = self._robot.data.root_link_state_w[:, 2]
            v = torch.norm(self._robot.data.root_com_lin_vel_b, dim=-1)

            twr_des = (self._wrench_des[:, 0] / self._robot_weight).cpu()
            twr_cmd = (self._thrust[:, 0, 2] / self._robot_weight).cpu()

            self.rp_history.append(torch.stack([roll_w, pitch_w]).cpu().numpy() * R2D)
            self.yaw_history.append(self.unwrapped_yaw.cpu().item() * R2D)
            self.z_history.append(z.cpu().numpy())
            self.v_history.append(v.cpu().numpy())
            self.twr_history.append(torch.stack([twr_des, twr_cmd], dim=-1).squeeze().numpy())
            self.actions_history.append(self._actions.squeeze(0).cpu().numpy())
            self.undelayed_actions_history.append(self._undelayed_actions.squeeze(0).cpu().numpy())

            omega_meas = self._robot.data.root_ang_vel_b
            roll_rate_des = (self.cfg.body_rate_scale_xy * self._actions[:, 1]).cpu().numpy()
            pitch_rate_des = (self.cfg.body_rate_scale_xy * self._actions[:, 2]).cpu().numpy()
            yaw_rate_des = (self.cfg.body_rate_scale_z * self._actions[:, 3]).cpu().numpy()

            self.roll_rate_history.append([roll_rate_des, omega_meas[:, 0].cpu().numpy()])
            self.pitch_rate_history.append([pitch_rate_des, omega_meas[:, 1].cpu().numpy()])
            self.yaw_rate_history.append([yaw_rate_des, omega_meas[:, 2].cpu().numpy()])

            if self.n_steps % PLOT_UPDATE_FREQ == 0:
                steps = np.arange(max(0, self.n_steps - self.max_len_deque), self.n_steps)

                rp_arr = np.array(self.rp_history)[-len(steps):]
                yaw_arr = np.array(self.yaw_history)[-len(steps):]
                z_arr = np.array(self.z_history)[-len(steps):]
                v_arr = np.array(self.v_history)[-len(steps):]
                twr_arr = np.array(self.twr_history)[-len(steps):]
                act_arr = np.array(self.actions_history)[-len(steps):]
                undel_act_arr = np.array(self.undelayed_actions_history)[-len(steps):]
                rr_arr = np.array(self.roll_rate_history)[-len(steps):]
                pr_arr = np.array(self.pitch_rate_history)[-len(steps):]
                yr_arr = np.array(self.yaw_rate_history)[-len(steps):]

                for i in range(2):
                    self.rp_lines[i].set_data(steps, rp_arr[:, i])
                self.yaw_line.set_data(steps, yaw_arr)
                self.z_line.set_data(steps, z_arr)
                self.v_line.set_data(steps, v_arr)

                for i in range(2):
                    self.twr_lines[i].set_data(steps, twr_arr[:, i])

                for i in range(self.cfg.action_space):
                    self.actions_lines[i].set_data(steps, act_arr[:, i])
                    self.undelayed_actions_lines[i].set_data(steps, undel_act_arr[:, i])

                for i in range(2):
                    self.body_rates_r_line[i].set_data(steps, rr_arr[:, i])
                    self.body_rates_p_line[i].set_data(steps, pr_arr[:, i])
                    self.body_rates_y_line[i].set_data(steps, yr_arr[:, i])

                for ax in list(self.data_axes) + list(self.body_rates_axes):
                    ax.relim()
                    ax.autoscale_view()

                self.data_fig.canvas.draw()
                self.data_fig.canvas.flush_events()
                self.body_rates_fig.canvas.draw()
                self.body_rates_fig.canvas.flush_events()

        return observations
