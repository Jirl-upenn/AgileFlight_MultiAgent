# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from pxr import Gf, UsdGeom, Sdf, UsdPhysics
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R

##
# Pre-defined configs
##
from isaaclab_assets.robots import CRAZYFLIE_CFG  # isort: skip

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)

# Adversary waypoint marker config (blue)
ADV_GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.05,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    }
)

def create_drone_config(prim_path: str, color: tuple):
    """Create a drone configuration with specified path and color.

    Args:
        prim_path: The prim path for the drone
        color: RGB color tuple for the drone (e.g., (1.0, 0.0, 0.0) for red)
    """
    return CRAZYFLIE_CFG.replace(
        prim_path=prim_path,
        collision_group=0,
        spawn=sim_utils.UsdFileCfg(
            usd_path=CRAZYFLIE_CFG.spawn.usd_path,
            activate_contact_sensors=True,
            visual_material=PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.2,
                roughness=0.5
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            )
        )
    )

# config for the ego drone's color (red)
CRAZYFLIE_CFG_COLORED = create_drone_config("/World/envs/env_.*/Robot", (1.0, 0.0, 0.0))

# config for the adversary drone's color (blue)
CRAZYFLIE_CFG_ADV_COLORED = create_drone_config("/World/envs/env_.*/Adversary", (0.0, 0.0, 1.0))



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
class QuadcopterEnvCfg(DirectMARLEnvCfg):
    # Define possible agents
    possible_agents = ["ego", "adversary"]

    # Define action spaces as dictionary
    action_spaces = {"ego": 4, "adversary": 4}

    # Wall configuration - set via command line argument
    use_wall: bool = True  # Default to True for backward compatibility

    # Track configuration - set via command line argument
    track: str = "complex"  # Options: "complex" or "lemniscate"

    # Define observation spaces as dictionary
    # These are dynamically set in __post_init__ based on use_wall
    # 42 without wall (no global pos), 45 with wall
    observation_spaces = {"ego": 45, "adversary": 45}

    # Define state space for centralized critic
    # Dynamically set in __post_init__ based on use_wall
    # (84) 42+42 without wall, (90) 45+45 with wall
    state_space = 90

    def __post_init__(self):
        """Update observation and state spaces based on use_wall configuration."""
        super().__post_init__()
        if self.use_wall:
            self.observation_spaces = {"ego": 45, "adversary": 45}
            self.state_space = 90
        else:
            self.observation_spaces = {"ego": 42, "adversary": 42}
            self.state_space = 84

    # env
    episode_length_s = 30.0             # episode_length = episode_length_s / dt / decimation
    action_space = 4
    observation_space = (
        3 +     # global position (only with wall)
         3 +     # linear velocity
         9 +     # attitude matrixf
        12 +     # relative desired position vertices waypoint 1
        12 +     # relative desired position vertices waypoint 2
        3 +    # adversary position in ego frame
        3  # adversary linear velocity in ego frame
    )  # Total: 42 without wall, 45 with wall (add 3 for global position)
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
        # disable_contact_processing=True,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0, replicate_physics=True)
    gate_model: GateModelCfg = field(default_factory=GateModelCfg)

    # robot (not used in _setup_scene, but kept for compatibility)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        force_threshold=0.0,
    )
    adv_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Adversary/body",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        force_threshold=0.0,
    )
    moment_scale = 0.01

    # Initialize variables
    beta = 1.0         # 1.0 for no smoothing, 0.0 for no update

    # Reset variables
    min_altitude = 0.1
    max_altitude = 3.0
    max_time_on_ground = 1.0

    # motor dynamics
    arm_length = 0.043
    k_eta = 2.3e-8
    k_m = 7.8e-10
    tau_m = 0.005
    motor_speed_min = 0.0
    motor_speed_max = 2500.0

    # PID Parameters
    # Roll/Pitch rate gains
    kp_omega_rp = 250.0
    ki_omega_rp = 500.0
    kd_omega_rp = 2.5
    i_limit_rp = 33.3
    
    # Yaw rate gains
    kp_omega_y = 120.0
    ki_omega_y = 16.70
    kd_omega_y = 0.0
    i_limit_y = 166.7
    body_rate_scale_xy = 100 * D2R
    body_rate_scale_z = 200 * D2R

    # Parameters from train.py or play.py
    is_train = None
    proximity_threshold = 0.05

    k_aero_xy = 9.1785e-7
    k_aero_z = 10.311e-7

    rewards = {}

class QuadcopterEnv(DirectMARLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        print(f">> QuadcopterEnv class path: {__file__}")

        super().__init__(cfg, render_mode, **kwargs)

        if len(cfg.rewards) > 0:
            self.rew = cfg.rewards
        elif self.cfg.is_train:
            raise ValueError("rewards not provided")

        # Initialize tensors for both drones
        self._init_ego_tensors()
        self._init_adversary_tensors()
        self._init_shared_state()

        # Track the last lap that received a bonus (to prevent duplicate bonuses)
        self._last_lap_bonus_claimed = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)

        # Initialize motor dynamics and control parameters
        self._init_motor_dynamics()
        self._init_logging()
        self._init_body_properties()
        self._init_randomization_ranges()

        self.set_debug_vis(self.cfg.debug_vis)
        self._initial_wp = 0

    def _init_drone_tensors(self, is_adversary=False):
        """Initialize tensors for a drone (ego or adversary).

        Args:
            is_adversary: If True, initialize tensors for adversary drone with '_adv' prefix.
                         If False, initialize tensors for ego drone with '_' prefix.
        """
        # Determine the prefix for attribute names
        prefix = "_adv" if is_adversary else ""

        # Action tensors
        setattr(self, f"{prefix}_actions", torch.zeros(self.num_envs, self.cfg.action_space, device=self.device))
        setattr(self, f"{prefix}_previous_actions", torch.zeros(self.num_envs, self.cfg.action_space, device=self.device))
        setattr(self, f"{prefix}_previous_yaw", torch.zeros(self.num_envs, device=self.device))

        # Control tensors
        setattr(self, f"{prefix}_thrust", torch.zeros(self.num_envs, 1, 3, device=self.device))
        setattr(self, f"{prefix}_moment", torch.zeros(self.num_envs, 1, 3, device=self.device))
        setattr(self, f"{prefix}_wrench_des", torch.zeros(self.num_envs, 4, device=self.device))
        setattr(self, f"{prefix}_motor_speeds", torch.zeros(self.num_envs, 4, device=self.device))
        setattr(self, f"{prefix}_motor_speeds_des", torch.zeros(self.num_envs, 4, device=self.device))
        setattr(self, f"{prefix}_previous_omega_meas", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_previous_omega_err", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_omega_err_integral", torch.zeros(self.num_envs, 3, device=self.device))

        # Navigation tensors
        setattr(self, f"{prefix}_desired_pos_w", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_last_distance_to_goal", torch.zeros(self.num_envs, device=self.device))
        setattr(self, f"{prefix}_n_laps", torch.zeros(self.num_envs, device=self.device))
        setattr(self, f"{prefix}_idx_wp", torch.zeros(self.num_envs, device=self.device, dtype=torch.int))
        setattr(self, f"{prefix}_n_gates_passed", torch.zeros(self.num_envs, device=self.device, dtype=torch.int))
        setattr(self, f"{prefix}_lap_completion_counted", torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        setattr(self, f"{prefix}_pose_drone_wrt_gate", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_prev_x_drone_wrt_gate", torch.ones(self.num_envs, device=self.device))

        # Crash tracking
        setattr(self, f"{prefix}_crashed", torch.zeros(self.num_envs, device=self.device, dtype=torch.int))
        # Special case: ego uses "_ego_was_crashed", adversary uses "_adv_was_crashed"
        crash_prefix = "_ego" if not is_adversary else "_adv"
        setattr(self, f"{crash_prefix}_was_crashed", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        setattr(self, f"{crash_prefix}_crash_times", torch.zeros(self.num_envs, dtype=torch.float32, device=self.device))
        setattr(self, f"{crash_prefix}_completed_crash_times", torch.empty(0, dtype=torch.float32, device=self.device))

        # Control parameters
        setattr(self, f"{prefix}_K_aero", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_kp_omega", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_ki_omega", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_kd_omega", torch.zeros(self.num_envs, 3, device=self.device))
        setattr(self, f"{prefix}_thrust_to_weight", torch.zeros(self.num_envs, device=self.device))

    def _init_ego_tensors(self):
        """Initialize tensors for ego drone."""
        self._init_drone_tensors(is_adversary=False)

    def _init_adversary_tensors(self):
        """Initialize tensors for adversary drone."""
        self._init_drone_tensors(is_adversary=True)

    def _init_shared_state(self):
        """Initialize shared state tensors."""
        # Termination flags for multi-agent environment
        self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.adv_reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.adv_reset_time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_motor_dynamics(self):
        """Initialize motor dynamics parameters."""
        # Motor dynamics
        self.cfg.thrust_to_weight = 3.15
        r2o2 = np.sqrt(2.0) / 2.0
        self._rotor_positions = torch.cat(
            [
                self.cfg.arm_length * torch.FloatTensor([[ r2o2,  r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[ r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[-r2o2, -r2o2, 0]]),
                self.cfg.arm_length * torch.FloatTensor([[-r2o2,  r2o2, 0]]),
            ],
            dim=0).to(self.device)
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

    def _init_logging(self):
        """Initialize logging tensors."""
        # Initialize episode sums for reward tracking (needed in both train and play modes)
        keys = []
        if hasattr(self, 'rew') and self.rew:
            keys = [key.split("_reward_scale")[0] for key in self.rew.keys() if key != "death_cost"]

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in keys
        }
        # Initialize adversary episode sums with the same structure
        self._adv_episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in keys
        }

        # Initialize tracking for total episode rewards (for min/mean/max logging)
        self._ego_episode_total_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._adv_episode_total_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # Use tensors for completed episode rewards to avoid numpy conversions
        self._ego_completed_episode_rewards = torch.zeros(0, dtype=torch.float, device=self.device)
        self._adv_completed_episode_rewards = torch.zeros(0, dtype=torch.float, device=self.device)
        # Track bonus episodes
        self._ego_bonus_episode_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._adv_bonus_episode_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._ego_completed_laps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._adv_completed_laps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def _init_body_properties(self):
        """Initialize body properties and physics parameters."""
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._adv_body_id = self._adversary.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self.inertia_tensor = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(-1, 3, 3).tile(self.num_envs, 1, 1).to(self.device)

    def _init_randomization_ranges(self):
        """Initialize randomization ranges for domain randomization."""
        if self.cfg.is_train:
            # TWR (needed for motor control)
            self._twr_min = self.cfg.thrust_to_weight
            self._twr_max = self.cfg.thrust_to_weight

            # Aerodynamics
            self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
            self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
            self._k_aero_z_min = self.cfg.k_aero_z * 0.5
            self._k_aero_z_max = self.cfg.k_aero_z * 2.0

            # PID gains - Roll/Pitch
            self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
            self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
            self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
            self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
            self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.7
            self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.2

            # PID gains - Yaw
            self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
            self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
            self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
            self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
            self._kd_omega_y_min = self.cfg.kd_omega_y * 0.7
            self._kd_omega_y_max = self.cfg.kd_omega_y * 1.2
        else:
            # TWR (needed for motor control)
            self._twr_min = self._twr_max = self.cfg.thrust_to_weight

            self._k_aero_xy_min = self._k_aero_xy_max = self.cfg.k_aero_xy
            self._k_aero_z_min = self._k_aero_z_max = self.cfg.k_aero_z

            # PID gains - Roll/Pitch
            self._kp_omega_rp_min = self._kp_omega_rp_max = self.cfg.kp_omega_rp
            self._ki_omega_rp_min = self._ki_omega_rp_max = self.cfg.ki_omega_rp
            self._kd_omega_rp_min = self._kd_omega_rp_max = self.cfg.kd_omega_rp

            # PID gains - Yaw
            self._kp_omega_y_min = self._kp_omega_y_max = self.cfg.kp_omega_y
            self._ki_omega_y_min = self._ki_omega_y_max = self.cfg.ki_omega_y
            self._kd_omega_y_min = self._kd_omega_y_max = self.cfg.kd_omega_y

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
            
            # Create adversary goal visualizer
            if not hasattr(self, "adv_goal_pos_visualizer"):
                adv_marker_cfg = ADV_GOAL_MARKER_CFG.copy()
                # -- adversary goal pose
                adv_marker_cfg.prim_path = "/Visuals/Command/adv_goal_position"
                self.adv_goal_pos_visualizer = VisualizationMarkers(adv_marker_cfg)
            # set their visibility to true
            self.adv_goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "adv_goal_pos_visualizer"):
                self.adv_goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        # update adversary markers
        if hasattr(self, "adv_goal_pos_visualizer"):
            self.adv_goal_pos_visualizer.visualize(self._adv_desired_pos_w)

    def _setup_scene(self):
        self._robot = Articulation(CRAZYFLIE_CFG_COLORED)
        self.scene.articulations["robot"] = self._robot

        self._adversary = Articulation(CRAZYFLIE_CFG_ADV_COLORED)
        self.scene.articulations["adversary"] = self._adversary
        
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        self._adv_contact_sensor = ContactSensor(self.cfg.adv_contact_sensor)
        self.scene.sensors["adv_contact_sensor"] = self._adv_contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # No collision filtering enabled, instead we use collision group 0 for all collisions local to the environment.
        # We also use collision group -1 for the terrain/ground plane for all environments to collide with
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        ##############
        ### TRACKS ###
        ##############

        # Select waypoints based on track configuration
        if self.cfg.track == "complex":
            self._waypoints = torch.tensor(
                [[1.5, 3.5, 0.75, 0.0, 0.0, -0.7854],
                 [-1.5, 3.5, 0.75, 0.0, 0.0, 0.7854],
                 [-2.0, -3.5, 2.0, 0.0, 0.0, 1.5708],
                 [-2.0, -3.5, 0.75, 0.0, 0.0, -1.5708],
                 [1.0, -1.0, 2.0, 0.0, 0.0, 3.1415],
                 [1.0, -3.5, 0.75, 0.0, 0.0, 0.0000]],
                device=self.device)
        elif self.cfg.track == "lemniscate":
            self._waypoints = torch.tensor(
                [[ 1.5, 3.50, 0.75, 0.0, 0.0, -1.57],
                 [ 0.0, 5.25, 1.50, 0.0, 0.0,  0.00],
                 [-2.0, 7.00, 0.75, 0.0, 0.0, -1.57],
                 [ 1.5, 7.00, 0.75, 0.0, 0.0,  1.57],
                 [ 0.0, 5.25, 1.50, 0.0, 0.0,  0.00],
                 [-2.0, 3.50, 0.75, 0.0, 0.0,  1.57]],
                device=self.device)
        else:
            raise ValueError(f"Unknown track type: {self.cfg.track}. Must be 'complex' or 'lemniscate'.")

        self._normal_vectors = torch.zeros(self._waypoints.shape[0], 3, device=self.device)
        self._waypoints_quat = torch.zeros(self._waypoints.shape[0], 4, device=self.device)

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

        # Setup walls if enabled (use config value from command line argument)
        self._use_wall = self.cfg.use_wall
        if self._use_wall:
            self._setup_walls()

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

            # 3.
            references_api_gate = model_scaled_ref_xform_prim.GetReferences()
            references_api_gate.AddReference(assetPath=model_usd_file_path)

            # 4. Apply physics
            # we need to apply collision to the mesh, not just the xform
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

    def _setup_walls(self):
        """Setup walls for track."""
        # Select walls based on track type: 1, 2, 3, 4 for complex track and 8, 9 for lemniscate
        if self.cfg.track == "complex":
            walls_set = (1, 2, 3, 4)
        else:  # lemniscate
            walls_set = (8, 9)

        # Define wall configurations
        wall_configs = {
            1: {"width": 2.0, "length": 0.1, "height": 8.0, "x": -2.0, "y": 2.0},
            2: {"width": 1.5, "length": 0.1, "height": 8.0, "x": 0.25, "y": -2.5},
            3: {"width": 4.0, "length": 0.1, "height": 8.0, "x": 3.0, "y": -0.25},
            4: {"width": 2.0, "length": 0.1, "height": 8.0, "x": 0.0, "y": 1.0},
            8: {"width": 0.2, "length": 1.3, "height": 8.0, "x": 0.0, "y": 7.65},
            9: {"width": 0.2, "length": 3.0, "height": 8.0, "x": -1.0, "y": 6.0, "rot_alpha": -0.78}
        }

        for wall_id in walls_set:
            if wall_id in wall_configs:
                config = wall_configs[wall_id]
                wall_center_z = config["height"] / 2.0

                # Build the wall configuration
                wall_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Wall_{wall_id}" if wall_id > 1 else "/World/envs/env_.*/Wall",
                    spawn=sim_utils.CuboidCfg(
                        size=(config["width"], config["length"], config["height"]),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(config["x"], config["y"], wall_center_z),
                        rot=(np.cos(config.get("rot_alpha", 0.0) / 2.0), 0.0, 0.0,
                             np.sin(config.get("rot_alpha", 0.0) / 2.0))
                    )
                )

                # Create and register the wall
                wall = RigidObject(wall_cfg)
                wall_name = f"wall_{wall_id}" if wall_id > 1 else "wall"
                self.scene.rigid_objects[wall_name] = wall

    def _compute_motor_speeds(self, wrench_des):
        f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        motor_speed_squared = f_des / self.cfg.k_eta
        motor_speeds_des = torch.sign(motor_speed_squared) * torch.sqrt(torch.abs(motor_speed_squared))
        motor_speeds_des = motor_speeds_des.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)

        return motor_speeds_des

    # without derivative kick
    def _get_moment_from_ctbr(self, actions, is_adversary=False):
        """Get moment commands from CTBR for either ego or adversary drone."""
        omega_des = torch.zeros(self.num_envs, 3, device=self.device)
        omega_des[:, :2] = self.cfg.body_rate_scale_xy * actions[:, 1:3]
        omega_des[:, 2] = self.cfg.body_rate_scale_z * actions[:, 3]

        # Select appropriate data based on agent type
        if is_adversary:
            omega_meas = self._adversary.data.root_ang_vel_b
            omega_err_integral = self._adv_omega_err_integral
            kp_omega = self._adv_kp_omega
            ki_omega = self._adv_ki_omega
            kd_omega = self._adv_kd_omega
            previous_omega_meas = self._adv_previous_omega_meas
        else:
            omega_meas = self._robot.data.root_ang_vel_b
            omega_err_integral = self._omega_err_integral
            kp_omega = self._kp_omega
            ki_omega = self._ki_omega
            kd_omega = self._kd_omega
            previous_omega_meas = self._previous_omega_meas

        omega_err = omega_des - omega_meas

        omega_err_integral += omega_err / self.cfg.pid_loop_rate_hz
        # Apply separate I-limits for roll/pitch and yaw
        if self.cfg.i_limit_rp > 0 or self.cfg.i_limit_y > 0:
            limits = torch.tensor(
                [self.cfg.i_limit_rp, self.cfg.i_limit_rp, self.cfg.i_limit_y],
                device=omega_err_integral.device
            )
            omega_err_integral = torch.clamp(
                omega_err_integral,
                min=-limits,
                max=limits
            )

        omega_int = omega_err_integral

        previous_omega_meas = torch.where(
            torch.abs(previous_omega_meas) < 0.0001,
            omega_meas,
            previous_omega_meas
        )
        omega_meas_dot = (omega_meas - previous_omega_meas) * self.cfg.pid_loop_rate_hz

        omega_dot = (
            kp_omega * omega_err +
            ki_omega * omega_int -
            kd_omega * omega_meas_dot
        )

        # Update state for the appropriate agent
        if is_adversary:
            self._adv_omega_err_integral = omega_err_integral
            self._adv_previous_omega_meas = omega_meas.clone()
        else:
            self._omega_err_integral = omega_err_integral
            self._previous_omega_meas = omega_meas.clone()

        cmd_moment = torch.bmm(self.inertia_tensor, omega_dot.unsqueeze(2)).squeeze(2)
        return cmd_moment

    ##########################################################
    ### Functions called in direct_rl_env.py in this order ###
    ##########################################################

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Handle actions from both agents"""
        ego_actions = actions["ego"]
        adv_actions = actions["adversary"]

        self._actions = ego_actions.clone().clamp(-1.0, 1.0)
        self._actions = self.cfg.beta * self._actions + (1 - self.cfg.beta) * self._previous_actions
        self._wrench_des[:, 0] = ((self._actions[:, 0] + 1.0) / 2.0) * self._robot_weight * self._thrust_to_weight
        self.pid_loop_counter = 0

        # apply action smoothing for adversary
        self._adv_actions = adv_actions.clone().clamp(-1.0, 1.0)
        self._adv_actions = self.cfg.beta * self._adv_actions + (1 - self.cfg.beta) * self._adv_previous_actions
        self._adv_wrench_des[:, 0] = ((self._adv_actions[:,
                                       0] + 1.0) / 2.0) * self._robot_weight * self._adv_thrust_to_weight
        self.adv_pid_loop_counter = 0

    def _apply_drone_action(self, is_adversary=False):
        """Apply action for either ego or adversary drone."""
        # Select appropriate data based on agent type
        if is_adversary:
            pid_loop_counter = self.adv_pid_loop_counter
            actions = self._adv_actions
            wrench_des = self._adv_wrench_des
            motor_speeds_des = self._adv_motor_speeds_des
            motor_speeds = self._adv_motor_speeds
            K_aero = self._adv_K_aero
            thrust = self._adv_thrust
            moment = self._adv_moment
            robot = self._adversary
            body_id = self._adv_body_id
        else:
            pid_loop_counter = self.pid_loop_counter
            actions = self._actions
            wrench_des = self._wrench_des
            motor_speeds_des = self._motor_speeds_des
            motor_speeds = self._motor_speeds
            K_aero = self._K_aero
            thrust = self._thrust
            moment = self._moment
            robot = self._robot
            body_id = self._body_id

        if pid_loop_counter % self.cfg.pid_loop_decimation == 0:
            wrench_des[:, 1:] = self._get_moment_from_ctbr(actions, is_adversary=is_adversary)
            motor_speeds_des_new = self._compute_motor_speeds(wrench_des)
            if is_adversary:
                self._adv_motor_speeds_des = motor_speeds_des_new
            else:
                self._motor_speeds_des = motor_speeds_des_new
            # Update local reference to use the new value
            motor_speeds_des = motor_speeds_des_new

        # Update counter
        if is_adversary:
            self.adv_pid_loop_counter += 1
        else:
            self.pid_loop_counter += 1

        motor_accel = (motor_speeds_des - motor_speeds) / self.cfg.tau_m
        motor_speeds += motor_accel * self.physics_dt
        motor_speeds = motor_speeds.clamp(self.cfg.motor_speed_min, self.cfg.motor_speed_max)
        motor_forces = self.cfg.k_eta * motor_speeds ** 2
        wrench = torch.matmul(motor_forces, self.f_to_TM.t())

        # Compute drag
        lin_vel_b = robot.data.root_com_lin_vel_b
        theta_dot = torch.sum(motor_speeds, dim=1, keepdim=True)
        drag = -theta_dot * K_aero.unsqueeze(0) * lin_vel_b

        thrust[:, 0, :] = drag
        thrust[:, 0, 2] += wrench[:, 0]
        moment[:, 0, :] = wrench[:, 1:]

        # Update motor speeds state
        if is_adversary:
            self._adv_motor_speeds = motor_speeds
        else:
            self._motor_speeds = motor_speeds

        robot.set_external_force_and_torque(thrust, moment, body_ids=body_id)

    def _apply_action(self):
        # Apply ego action
        self._apply_drone_action(is_adversary=False)
        # Apply adversary action
        self._apply_drone_action(is_adversary=True)

    def _get_dones_helper(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Compute ego pose
        self._pose_drone_wrt_gate, _ = subtract_frame_transforms(
            self._waypoints[self._idx_wp, :3] + self._terrain.env_origins,
            self._waypoints_quat[self._idx_wp, :],
            self._robot.data.root_link_state_w[:, :3]
        )
        # Compute adversary pose
        self._adv_pose_drone_wrt_gate, _ = subtract_frame_transforms(
            self._waypoints[self._adv_idx_wp, :3] + self._terrain.env_origins,
            self._waypoints_quat[self._adv_idx_wp, :],
            self._adversary.data.root_link_state_w[:, :3]
        )

        # ----------------- Ego and Adv Termination -----------------
        # Check collision between ego and adversary drones (0.2m radius)
        ego_pos = self._robot.data.root_link_pos_w
        adv_pos = self._adversary.data.root_link_pos_w
        drone_distance = torch.linalg.norm(ego_pos - adv_pos, dim=1)

        def check_death_conditions(pose_wrt_gate, pos_w, quat_w, episode_time, idx_wp, prev_x_wrt_gate):
            # Check if drone is inside the gate boundaries
            cond_gate_inside = (torch.abs(pose_wrt_gate[:, 1]) < self._gate_model_cfg_data.gate_side / 2 - 0.1) & \
                               (torch.abs(pose_wrt_gate[:, 2]) < self._gate_model_cfg_data.gate_side / 2 - 0.1)
            
            # Check if drone passed through the gate
            cond_gate_through = (pose_wrt_gate[:, 0] < 0.0) & (prev_x_wrt_gate > 0.0)
            # Collision with gate if passed through but not inside gate boundaries
            cond_gate = cond_gate_through & ~cond_gate_inside

            # Altitude conditions
            cond_h_min_time = (pos_w[:, 2] < self.cfg.min_altitude) & (episode_time > self.cfg.max_time_on_ground)
            cond_max_h = pos_w[:, 2] > self.cfg.max_altitude

            # Boundary conditions (flybox) for x and y positions based on track type
            if self.cfg.track == "complex":
                cond_x_boundary = (pos_w[:, 0] > 5.0) | (pos_w[:, 0] < -6.0)  # x must be between -6 and +5
                cond_y_boundary = (torch.abs(pos_w[:, 1]) > 9.0)  # +/- 9 in y direction
            else:  # lemniscate
                cond_x_boundary = (pos_w[:, 0] > 2.5) | (pos_w[:, 0] < -2.5)  # x must be between -2.5 and +2.5
                cond_y_boundary = (pos_w[:, 1] > 9.0) | (pos_w[:, 1] < 2.0)  # y must be between +2 and +9

            # Flip conditions
            rpy = euler_xyz_from_quat(quat_w)
            roll = wrap_to_pi(rpy[0])
            pitch = wrap_to_pi(rpy[1])
            max_angle = 360 * D2R  # Changed from 180 to 360 to match test_lorenzo
            cond_flip = (torch.abs(roll) > max_angle) | (torch.abs(pitch) > max_angle)

            return cond_flip | cond_h_min_time | cond_max_h | cond_gate | cond_x_boundary | cond_y_boundary

        episode_time = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation
        
        # Add crash condition: if stuck (crashed) for more than 100 timesteps
        cond_crashed = self._crashed > 100
        cond_adv_crashed = self._adv_crashed > 100

        ego_died = check_death_conditions(
            self._pose_drone_wrt_gate,
            self._robot.data.root_link_pos_w,
            self._robot.data.root_quat_w,
            episode_time,
            self._idx_wp,
            self._prev_x_drone_wrt_gate
        ) | cond_crashed

        adv_died = check_death_conditions(
            self._adv_pose_drone_wrt_gate,
            self._adversary.data.root_link_pos_w,
            self._adversary.data.root_quat_w,
            episode_time,
            self._adv_idx_wp,
            self._adv_prev_x_drone_wrt_gate
        ) | cond_adv_crashed

        ego_timeout = self.episode_length_buf >= self.max_episode_length - 1
        adv_timeout = ego_timeout.clone()  # can be agent-specific if needed

        # Return stuck status as third dictionary (stuck for >100 timesteps only)
        stuck_dict = {
            "ego": cond_crashed,
            "adversary": cond_adv_crashed
        }

        return {
            "ego": ego_died,
            "adversary": adv_died
        }, {
            "ego": ego_timeout,
            "adversary": adv_timeout
        }, stuck_dict


    def _calculate_drone_rewards(self, is_adversary=False) -> torch.Tensor:
        """Calculate rewards for either ego or adversary drone.

        Args:
            is_adversary: If True, calculate for adversary drone. If False, calculate for ego drone.

        Returns:
            Tensor containing calculated rewards for the specified drone.
        """
        # Select appropriate tensors based on drone type
        if is_adversary:
            robot = self._adversary
            contact_sensor = self._adv_contact_sensor
            actions = self._adv_actions
            pose_drone_wrt_gate = self._adv_pose_drone_wrt_gate
            prev_x_drone_wrt_gate = self._adv_prev_x_drone_wrt_gate
            idx_wp = self._adv_idx_wp
            n_gates_passed = self._adv_n_gates_passed
            desired_pos_w = self._adv_desired_pos_w
            last_distance_to_goal = self._adv_last_distance_to_goal
            crashed = self._adv_crashed
            episode_sums = self._adv_episode_sums
            episode_total_rewards = self._adv_episode_total_rewards
            completed_laps = self._adv_completed_laps
            lap_completion_counted = self._adv_lap_completion_counted
            bonus_episode_count = self._adv_bonus_episode_count
            was_crashed = self._adv_was_crashed
            reset_terminated = self.adv_reset_terminated
        else:
            robot = self._robot
            contact_sensor = self._contact_sensor
            actions = self._actions
            pose_drone_wrt_gate = self._pose_drone_wrt_gate
            prev_x_drone_wrt_gate = self._prev_x_drone_wrt_gate
            idx_wp = self._idx_wp
            n_gates_passed = self._n_gates_passed
            desired_pos_w = self._desired_pos_w
            last_distance_to_goal = self._last_distance_to_goal
            crashed = self._crashed
            episode_sums = self._episode_sums
            episode_total_rewards = self._ego_episode_total_rewards
            completed_laps = self._ego_completed_laps
            lap_completion_counted = self._lap_completion_counted
            bonus_episode_count = self._ego_bonus_episode_count
            was_crashed = self._ego_was_crashed
            reset_terminated = self.reset_terminated

        lap_completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Check to change setpoint
        dist_to_gate = torch.linalg.norm(pose_drone_wrt_gate, dim=1)
        gate_passed = (dist_to_gate < 1.0) & \
                      (pose_drone_wrt_gate[:, 0] < 0.0) & \
                      (prev_x_drone_wrt_gate > 0.0) & \
                      (torch.abs(pose_drone_wrt_gate[:, 1]) < self._gate_model_cfg_data.gate_side / 2) & \
                      (torch.abs(pose_drone_wrt_gate[:, 2]) < self._gate_model_cfg_data.gate_side / 2)

        # Update previous x position for next iteration
        if is_adversary:
            self._adv_prev_x_drone_wrt_gate = pose_drone_wrt_gate[:, 0].clone()
        else:
            self._prev_x_drone_wrt_gate = pose_drone_wrt_gate[:, 0].clone()

        ids_close = torch.where(gate_passed)[0]

        n_gates_passed[ids_close] += 1

        idx_wp[ids_close] = (idx_wp[ids_close] + 1) % self._waypoints.shape[0]
        lap_completed[ids_close] = (n_gates_passed[ids_close] > self._waypoints.shape[0]) & \
                                   ((n_gates_passed[ids_close] % self._waypoints.shape[0]) == 1)

        desired_pos_w[ids_close, :2] = self._waypoints[idx_wp[ids_close], :2]
        desired_pos_w[ids_close, :2] += self._terrain.env_origins[ids_close, :2]
        desired_pos_w[ids_close, 2] = self._waypoints[idx_wp[ids_close], 2]

        distance_to_goal = torch.linalg.norm(desired_pos_w - robot.data.root_link_pos_w, dim=1)
        last_distance_to_goal[ids_close] = 1.05 * distance_to_goal[ids_close].clone()

        drone_pos = robot.data.root_link_pos_w

        # Collision detection using contact sensor
        contact_forces = contact_sensor.data.net_forces_w
        crashed_now = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.episode_length_buf > 100).int()
        if is_adversary:
            self._adv_crashed = self._adv_crashed + crashed_now * mask
        else:
            self._crashed = self._crashed + crashed_now * mask

        if self.cfg.is_train:
            # Gate passing reward: give bonus when drone passes a gate
            gate_pass_bonus = torch.zeros(self.num_envs, device=self.device)
            gate_pass_bonus[ids_close] = 1.0  # Base reward for passing through a gate

            roll_pitch_rates = torch.sum(torch.square(actions[:, 1:3]), dim=1)
            yaw_rate = torch.square(actions[:, 3])

            # Determine which drone is leading
            ego_wp = self._n_gates_passed
            adv_wp = self._adv_n_gates_passed
            ego_dist = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
            adv_dist = torch.linalg.norm(self._adv_desired_pos_w - self._adversary.data.root_link_pos_w, dim=1)

            # Determine if current drone is leading:
            # 1. If has more gates passed → leading
            # 2. If tied on gates → closer to next gate = leading
            if is_adversary:
                is_leading = torch.where(
                    adv_wp > ego_wp, torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
                    torch.where(adv_wp < ego_wp, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
                               adv_dist < ego_dist)  # When tied on gates, closer drone leads
                )
            else:
                is_leading = torch.where(
                    ego_wp > adv_wp, torch.ones(self.num_envs, device=self.device, dtype=torch.bool),
                    torch.where(ego_wp < adv_wp, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
                               ego_dist < adv_dist)  # When tied on gates, closer drone leads
                )

            # Add lap completion bonus (+100 when returning to first index)
            lap_bonus = torch.zeros(self.num_envs, device=self.device)
            current_lap = n_gates_passed // self._waypoints.shape[0]  # Calculate which lap we're on
            lap_completed = (n_gates_passed > self._waypoints.shape[0]) & (idx_wp == 1)

            # Give bonus only if:
            # 1. Lap is completed
            # 2. Drone is leading
            # 3. This lap hasn't given bonus yet
            eligible_for_bonus = lap_completed & is_leading & (current_lap > self._last_lap_bonus_claimed)

            lap_bonus[eligible_for_bonus] = 100.0
            # Update the last lap that gave bonus (shared tracker)
            self._last_lap_bonus_claimed[eligible_for_bonus] = current_lap[eligible_for_bonus]
            # Track bonus episodes
            bonus_episode_count[eligible_for_bonus] += 1

            # Track lap completions (independent of bonus)
            new_lap_completion_mask = lap_completed & ~lap_completion_counted
            completed_laps[new_lap_completion_mask] += 1
            lap_completion_counted[new_lap_completion_mask] = True
            # Reset lap completion flag when agent is not at gate index 1
            not_at_gate_1 = idx_wp != 1
            lap_completion_counted[not_at_gate_1] = False

            # Adjust gate pass bonus based on competition status
            # Leading drone gets full bonus (1.0)
            # Losing drone gets half bonus (0.5)
            gate_pass_bonus = torch.where(
                (gate_pass_bonus > 0) & ~is_leading,
                gate_pass_bonus * 0.5,
                gate_pass_bonus
            )

            rewards = {
                "gate_pass": gate_pass_bonus * self.rew['gate_pass_reward_scale'],
                "roll_pitch_rates": roll_pitch_rates * self.rew['roll_pitch_rates_reward_scale'] * self.step_dt,
                "yaw_rate": yaw_rate * self.rew["yaw_rate_reward_scale"] * self.step_dt,
                "lap_bonus": lap_bonus * self.rew['lap_bonus_reward_scale'],
                "crash": crashed_now * self.rew['crash_reward_scale'],  # separate from the death penalty
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            # Apply crash penalty when transitioning from non-crashed to crashed state
            new_crash = reset_terminated & ~was_crashed
            crash_penalty = torch.zeros_like(reward)
            crash_penalty[new_crash] = self.rew['death_cost']
            reward = reward + crash_penalty

            # Update the previous crashed state for next step
            if is_adversary:
                self._adv_was_crashed = reset_terminated.clone()
            else:
                self._ego_was_crashed = reset_terminated.clone()

            # Logging
            for key, value in rewards.items():
                episode_sums[key] += value

            # Track total episode rewards
            episode_total_rewards += reward
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def _get_ego_rewards(self) -> torch.Tensor:
        """Calculate rewards for ego drone."""
        return self._calculate_drone_rewards(is_adversary=False)

    def _get_adv_rewards(self) -> torch.Tensor:
        """Calculate rewards for adversary drone."""
        return self._calculate_drone_rewards(is_adversary=True)
    
    def _track_episode_statistics(self, env_ids: torch.Tensor):
        """Track completed episode rewards and crash times for min/mean/max statistics."""
        if len(env_ids) > 0:
            # Store ego completed episode rewards using tensor concatenation
            self._ego_completed_episode_rewards = torch.cat([
                self._ego_completed_episode_rewards,
                self._ego_episode_total_rewards[env_ids]
            ])
            # Store adversary completed episode rewards
            self._adv_completed_episode_rewards = torch.cat([
                self._adv_completed_episode_rewards,
                self._adv_episode_total_rewards[env_ids]
            ])

            # Store crash times for crashed episodes (not timeouts)
            ego_crash_mask = self.reset_terminated[env_ids] & ~self.reset_time_outs[env_ids]
            adv_crash_mask = self.adv_reset_terminated[env_ids] & ~self.adv_reset_time_outs[env_ids]

            if ego_crash_mask.any():
                self._ego_completed_crash_times = torch.cat([
                    self._ego_completed_crash_times,
                    self._ego_crash_times[env_ids[ego_crash_mask]]
                ])

            if adv_crash_mask.any():
                self._adv_completed_crash_times = torch.cat([
                    self._adv_completed_crash_times,
                    self._adv_crash_times[env_ids[adv_crash_mask]]
                ])

            # Keep only last 100 episodes for memory efficiency
            if len(self._ego_completed_episode_rewards) > 100:
                self._ego_completed_episode_rewards = self._ego_completed_episode_rewards[-100:]
            if len(self._adv_completed_episode_rewards) > 100:
                self._adv_completed_episode_rewards = self._adv_completed_episode_rewards[-100:]
            if len(self._ego_completed_crash_times) > 100:
                self._ego_completed_crash_times = self._ego_completed_crash_times[-100:]
            if len(self._adv_completed_crash_times) > 100:
                self._adv_completed_crash_times = self._adv_completed_crash_times[-100:]

    def _log_episode_metrics(self, env_ids: torch.Tensor):
        """Log episode rewards, termination stats, and bonus rates."""
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Add ego total reward min/mean/max if we have completed episodes
        if len(self._ego_completed_episode_rewards) > 0:
            extras["Ego_Total_Reward/min"] = torch.min(self._ego_completed_episode_rewards)
            extras["Ego_Total_Reward/mean"] = torch.mean(self._ego_completed_episode_rewards)
            extras["Ego_Total_Reward/max"] = torch.max(self._ego_completed_episode_rewards)

        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        # Add crash time statistics
        if len(self._ego_completed_crash_times) > 0:
            extras["Ego_Crash_Time/min"] = torch.min(self._ego_completed_crash_times).item()
            extras["Ego_Crash_Time/mean"] = torch.mean(self._ego_completed_crash_times).item()
            extras["Ego_Crash_Time/max"] = torch.max(self._ego_completed_crash_times).item()

        if len(self._adv_completed_crash_times) > 0:
            extras["Adv_Crash_Time/min"] = torch.min(self._adv_completed_crash_times).item()
            extras["Adv_Crash_Time/mean"] = torch.mean(self._adv_completed_crash_times).item()
            extras["Adv_Crash_Time/max"] = torch.max(self._adv_completed_crash_times).item()

        self.extras["log"].update(extras)

        # Log adversary episode rewards
        adv_extras = dict()
        for key in self._adv_episode_sums.keys():
            adv_episodic_sum_avg = torch.mean(self._adv_episode_sums[key][env_ids])
            adv_extras["Adversary_Episode_Reward/" + key] = adv_episodic_sum_avg / self.max_episode_length_s
            self._adv_episode_sums[key][env_ids] = 0.0

        # Add adversary total reward min/mean/max if we have completed episodes
        if len(self._adv_completed_episode_rewards) > 0:
            adv_extras["Adversary_Total_Reward/min"] = torch.min(self._adv_completed_episode_rewards)
            adv_extras["Adversary_Total_Reward/mean"] = torch.mean(self._adv_completed_episode_rewards)
            adv_extras["Adversary_Total_Reward/max"] = torch.max(self._adv_completed_episode_rewards)

        self.extras["log"].update(adv_extras)

        # Calculate and log bonus episode rates
        if len(env_ids) > 0:
            # Calculate bonus episode rate for ego agents
            ego_bonus_episodes = self._ego_bonus_episode_count[env_ids].float()
            ego_completed_laps = self._ego_completed_laps[env_ids].float()
            ego_bonus_rate = torch.where(ego_completed_laps > 0,
                                        ego_bonus_episodes / ego_completed_laps,
                                        torch.zeros_like(ego_bonus_episodes))

            # Calculate bonus episode rate for adversary agents
            adv_bonus_episodes = self._adv_bonus_episode_count[env_ids].float()
            adv_completed_laps = self._adv_completed_laps[env_ids].float()
            adv_bonus_rate = torch.where(adv_completed_laps > 0,
                                        adv_bonus_episodes / adv_completed_laps,
                                        torch.zeros_like(adv_bonus_episodes))

            # Log bonus rates
            bonus_extras = dict()
            bonus_extras["Ego_Bonus_Rate/mean"] = torch.mean(ego_bonus_rate)
            bonus_extras["Adversary_Bonus_Rate/mean"] = torch.mean(adv_bonus_rate)

            self.extras["log"].update(bonus_extras)

            # Reset bonus tracking for resetting environments
            self._ego_bonus_episode_count[env_ids] = 0
            self._adv_bonus_episode_count[env_ids] = 0
            self._ego_completed_laps[env_ids] = 0
            self._adv_completed_laps[env_ids] = 0

        # Reset episode total rewards for environments that are resetting
        self._ego_episode_total_rewards[env_ids] = 0.0
        self._adv_episode_total_rewards[env_ids] = 0.0

    def _reset_ego_state(self, env_ids: torch.Tensor, n_reset: int):
        """Reset ego drone state, position, and parameters.
        Returns waypoint_indices and default_root_state for use by adversary reset."""
        # Reset ego action and motor states
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_yaw[env_ids] = 0.0
        self._motor_speeds[env_ids] = 0.0
        self._previous_omega_meas[env_ids] = 0.0
        self._previous_omega_err[env_ids] = 0.0
        self._omega_err_integral[env_ids] = 0.0

        # Reset crashed state tracking for new episodes
        self._ego_was_crashed[env_ids] = False

        # Reset joints state
        joint_pos = self._robot.data.default_joint_pos[env_ids]     # not important
        joint_vel = self._robot.data.default_joint_vel[env_ids]     #
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self._robot.data.default_root_state[env_ids]   # [pos, quat, lin_vel, ang_vel] in local environment frame. Shape is (num_instances, 13)
        # Always use default reset logic (ResetBuffer removed)
        waypoint_indices = torch.randint(0, self._waypoints.shape[0], (n_reset,), device=self.device, dtype=self._idx_wp.dtype)

        # Get random starting poses behind waypoints
        x0_wp = self._waypoints[waypoint_indices][:, 0]
        y0_wp = self._waypoints[waypoint_indices][:, 1]
        theta = self._waypoints[waypoint_indices][:, -1]

        x_local = torch.empty(n_reset, device=self.device).uniform_(-2.0, -0.5)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot

        # Reset robots state
        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        z_wp = self._waypoints[waypoint_indices][:, 2]
        default_root_state[:, 2] = torch.empty(n_reset, device=self.device).uniform_(-0.5, 0.5) + z_wp
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        initial_yaw = torch.atan2(-initial_y, -initial_x)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=self.device),
            torch.zeros(1, device=self.device),
            initial_yaw.to(self.device) + torch.empty(1, device=self.device).uniform_(-0.15, 0.15)
        )
        default_root_state[:, 3:7] = quat

        # Fixed percentage of ground resets. Overwrites 10% of environments with ground resets from initial wp
        percent_ground = 0.1

        ground_mask = torch.rand(n_reset, device=self.device) < percent_ground
        ground_local_ids = torch.nonzero(ground_mask, as_tuple=False).squeeze(-1)

        if ground_local_ids.numel() > 0:
            waypoint_indices[ground_local_ids] = self._initial_wp
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

        if not self.cfg.is_train:
            # Initial position during play
            x0 = 1.815
            y0 = 2.890

            x0 = 2.89
            z0 = 0.1
            yaw0 = torch.tensor([179.422 * D2R], device=self.device)

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

            waypoint_indices = self._initial_wp

        self._idx_wp[env_ids] = waypoint_indices

        self._desired_pos_w[env_ids, :2] = self._waypoints[waypoint_indices, :2].clone()
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = self._waypoints[waypoint_indices, 2].clone()

        self._last_distance_to_goal[env_ids] = torch.linalg.norm(self._desired_pos_w[env_ids, :2] - self._robot.data.root_link_pos_w[env_ids, :2], dim=1)
        self._n_gates_passed[env_ids] = 0
        self._lap_completion_counted[env_ids] = False
        self._last_lap_bonus_claimed[env_ids] = 0

        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self._n_laps[env_ids] = 0

        self._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(self._waypoints[self._idx_wp[env_ids], :3] + self._terrain.env_origins[env_ids, :3],
                                                                          self._waypoints_quat[self._idx_wp[env_ids], :],
                                                                          self._robot.data.root_link_state_w[env_ids, :3])
        self._prev_x_drone_wrt_gate[env_ids] = torch.ones(len(env_ids), device=self.device)

        # Reset crashed counters
        self._crashed[env_ids] = 0

        self._K_aero[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(self._k_aero_xy_min, self._k_aero_xy_max)
        self._K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(self._k_aero_z_min, self._k_aero_z_max)

        # Reset PID gains for roll/pitch and yaw using 3D tensors
        kp_omega_rp = torch.empty(n_reset, device=self.device).uniform_(self._kp_omega_rp_min, self._kp_omega_rp_max)
        ki_omega_rp = torch.empty(n_reset, device=self.device).uniform_(self._ki_omega_rp_min, self._ki_omega_rp_max)
        kd_omega_rp = torch.empty(n_reset, device=self.device).uniform_(self._kd_omega_rp_min, self._kd_omega_rp_max)

        kp_omega_y = torch.empty(n_reset, device=self.device).uniform_(self._kp_omega_y_min, self._kp_omega_y_max)
        ki_omega_y = torch.empty(n_reset, device=self.device).uniform_(self._ki_omega_y_min, self._ki_omega_y_max)
        kd_omega_y = torch.empty(n_reset, device=self.device).uniform_(self._kd_omega_y_min, self._kd_omega_y_max)

        # Stack into 3D tensors: [roll_gain, pitch_gain, yaw_gain]
        self._kp_omega[env_ids] = torch.stack([kp_omega_rp, kp_omega_rp, kp_omega_y], dim=1)
        self._ki_omega[env_ids] = torch.stack([ki_omega_rp, ki_omega_rp, ki_omega_y], dim=1)
        self._kd_omega[env_ids] = torch.stack([kd_omega_rp, kd_omega_rp, kd_omega_y], dim=1)

        self._thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(self._twr_min, self._twr_max)

        return waypoint_indices, default_root_state

    def _reset_adversary_state(self, env_ids: torch.Tensor, n_reset: int, waypoint_indices: torch.Tensor, default_root_state: torch.Tensor):
        """Reset adversary drone state, position, and parameters."""
        # Reset adversary crashed state tracking
        self._adv_was_crashed[env_ids] = False
        # Reset adversary crashed counter
        self._adv_crashed[env_ids] = 0

        # reset adversary state
        self._adv_actions[env_ids] = 0.0
        self._adv_previous_actions[env_ids] = 0.0
        self._adv_previous_yaw[env_ids] = 0.0
        self._adv_n_laps[env_ids] = 0
        self._adv_motor_speeds[env_ids] = 0.0
        self._adv_previous_omega_err[env_ids] = 0.0
        self._adv_omega_err_integral[env_ids] = 0.0
        self._adv_previous_omega_meas[env_ids] = 0.0

        # Reset adversary joints state (for completeness, matching ego)
        adv_joint_pos = self._adversary.data.default_joint_pos[env_ids]
        adv_joint_vel = self._adversary.data.default_joint_vel[env_ids]
        self._adversary.write_joint_state_to_sim(adv_joint_pos, adv_joint_vel, None, env_ids)

        # Reset adversary robot pose (ResetBuffer removed, always using default logic)
        adv_default_root_state = self._adversary.data.default_root_state[env_ids]

        # Position adversary randomly within 1 meter of ego
        # Sample random angle and distance
        angle = torch.empty(n_reset, device=self.device).uniform_(0, 2 * torch.pi)
        distance = torch.empty(n_reset, device=self.device).uniform_(0.1, 1.0)  # Between 0.1 and 1 meter

        # Calculate X and Y offsets based on angle and distance
        adv_default_root_state[:, 0] = default_root_state[:, 0] + distance * torch.cos(angle)
        adv_default_root_state[:, 1] = default_root_state[:, 1] + distance * torch.sin(angle)
        adv_default_root_state[:, 2] = default_root_state[:, 2]  # Same height
        adv_default_root_state[:, 3:7] = default_root_state[:, 3:7]  # Same orientation

        # Set adversary waypoints (same as ego for now)
        self._adv_idx_wp[env_ids] = waypoint_indices
        self._adv_desired_pos_w[env_ids, :2] = self._waypoints[waypoint_indices, :2].clone()
        self._adv_desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._adv_desired_pos_w[env_ids, 2] = self._waypoints[waypoint_indices, 2].clone()

        self._adversary.write_root_link_pose_to_sim(adv_default_root_state[:, :7], env_ids)
        self._adversary.write_root_com_velocity_to_sim(adv_default_root_state[:, 7:], env_ids)

        # Initialize adversary last distance to goal
        self._adv_last_distance_to_goal[env_ids] = torch.linalg.norm(
            self._adv_desired_pos_w[env_ids, :2] - self._adversary.data.root_link_pos_w[env_ids, :2],
            dim=1
        )

        # Reset adversary gates passed counter
        self._adv_n_gates_passed[env_ids] = 0
        self._adv_lap_completion_counted[env_ids] = False

        # Initialize adversary pose relative to gate
        self._adv_pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self._waypoints[self._adv_idx_wp[env_ids], :3] + self._terrain.env_origins[env_ids, :3],
            self._waypoints_quat[self._adv_idx_wp[env_ids], :],
            self._adversary.data.root_link_state_w[env_ids, :3]
        )
        self._adv_prev_x_drone_wrt_gate[env_ids] = torch.ones(len(env_ids), device=self.device)

        # Reset adversary aerodynamic parameters
        self._adv_K_aero[env_ids, :2] = torch.empty(n_reset, 2, device=self.device).uniform_(
            self._k_aero_xy_min, self._k_aero_xy_max
        )
        self._adv_K_aero[env_ids, 2] = torch.empty(n_reset, device=self.device).uniform_(
            self._k_aero_z_min, self._k_aero_z_max
        )

        # Reset adversary control gains using 3D tensors
        adv_kp_omega_rp = torch.empty(n_reset, device=self.device).uniform_(
            self._kp_omega_rp_min, self._kp_omega_rp_max
        )
        adv_ki_omega_rp = torch.empty(n_reset, device=self.device).uniform_(
            self._ki_omega_rp_min, self._ki_omega_rp_max
        )
        adv_kd_omega_rp = torch.empty(n_reset, device=self.device).uniform_(
            self._kd_omega_rp_min, self._kd_omega_rp_max
        )

        adv_kp_omega_y = torch.empty(n_reset, device=self.device).uniform_(
            self._kp_omega_y_min, self._kp_omega_y_max
        )
        adv_ki_omega_y = torch.empty(n_reset, device=self.device).uniform_(
            self._ki_omega_y_min, self._ki_omega_y_max
        )
        adv_kd_omega_y = torch.empty(n_reset, device=self.device).uniform_(
            self._kd_omega_y_min, self._kd_omega_y_max
        )

        # Stack into 3D tensors: [roll_gain, pitch_gain, yaw_gain]
        self._adv_kp_omega[env_ids] = torch.stack([adv_kp_omega_rp, adv_kp_omega_rp, adv_kp_omega_y], dim=1)
        self._adv_ki_omega[env_ids] = torch.stack([adv_ki_omega_rp, adv_ki_omega_rp, adv_ki_omega_y], dim=1)
        self._adv_kd_omega[env_ids] = torch.stack([adv_kd_omega_rp, adv_kd_omega_rp, adv_kd_omega_y], dim=1)

        # Reset adversary thrust-to-weight ratio
        self._adv_thrust_to_weight[env_ids] = torch.empty(n_reset, device=self.device).uniform_(
            self._twr_min, self._twr_max
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        if self.cfg.is_train:
            self._track_episode_statistics(env_ids)
            self._log_episode_metrics(env_ids)

        self._robot.reset(env_ids)
        self._adversary.reset(env_ids)
        super()._reset_idx(env_ids)

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset ego drone state and get waypoint indices and root state for adversary
        waypoint_indices, default_root_state = self._reset_ego_state(env_ids, n_reset)

        # Reset adversary state using the waypoint indices and root state from ego
        self._reset_adversary_state(env_ids, n_reset, waypoint_indices, default_root_state)


    def _get_drone_observations(self, is_adversary=False) -> dict:
        """Get observations for either ego or adversary drone.

        Args:
            is_adversary: If True, get observations for adversary drone. If False, get observations for ego drone.

        Returns:
            Dictionary containing observations with appropriate key ("policy" or "adv_policy").
        """
        # Select appropriate robot and indices based on drone type
        if is_adversary:
            robot = self._adversary
            other_robot = self._robot
            idx_wp = self._adv_idx_wp
            previous_yaw = self._adv_previous_yaw
            n_laps = self._adv_n_laps
            obs_key = "adv_policy"
        else:
            robot = self._robot
            other_robot = self._adversary
            idx_wp = self._idx_wp
            previous_yaw = self._previous_yaw
            n_laps = self._n_laps
            obs_key = "policy"

        # Compute waypoint indices
        curr_idx = idx_wp % self._waypoints.shape[0]
        next_idx = (idx_wp + 1) % self._waypoints.shape[0]

        # Get waypoint positions and orientations
        wp_curr_pos = self._waypoints[curr_idx, :3]
        wp_next_pos = self._waypoints[next_idx, :3]
        quat_curr = self._waypoints_quat[curr_idx]
        quat_next = self._waypoints_quat[next_idx]

        # Compute rotation matrices
        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        # Transform waypoint vertices to world frame
        verts_curr = torch.bmm(self._local_square, rot_curr.transpose(1, 2)) + wp_curr_pos.unsqueeze(1) + self._terrain.env_origins.unsqueeze(1)
        verts_next = torch.bmm(self._local_square, rot_next.transpose(1, 2)) + wp_next_pos.unsqueeze(1) + self._terrain.env_origins.unsqueeze(1)

        # Transform waypoints to body frame of current robot
        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3))
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3))

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        # Get attitude information
        quat_w = robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        # Get other robot's position relative to current robot in current robot's body frame
        other_pos_body_frame, _ = subtract_frame_transforms(
            robot.data.root_link_state_w[:, :3],  # current robot position
            robot.data.root_link_state_w[:, 3:7],  # current robot orientation
            other_robot.data.root_link_pos_w  # other robot position
        )

        # Transform other robot's velocity to current robot's body frame
        other_lin_vel_w = other_robot.data.root_com_lin_vel_w
        other_lin_vel_body_frame = torch.bmm(
            attitude_mat.transpose(1, 2),  # world to body rotation
            other_lin_vel_w.unsqueeze(2)  # [N, 3, 1]
        ).squeeze(2)  # [N, 3]

        # Build observation list
        obs_list = []
        if self._use_wall:
            obs_list.append(robot.data.root_link_pos_w)  # global position (3) - only with wall
        obs_list.extend([
            robot.data.root_com_lin_vel_b,  # linear velocity (3)
            attitude_mat.view(attitude_mat.shape[0], -1),  # attitude matrix (9)
            waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),  # waypoint 1 (12)
            waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),  # waypoint 2 (12)
            other_pos_body_frame,  # other robot position in body frame (3)
            other_lin_vel_body_frame,  # other robot velocity in body frame (3)
        ])
        obs = torch.cat(obs_list, dim=-1)

        # Update yaw tracking and lap counts
        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - previous_yaw
        n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        # Update state variables
        if is_adversary:
            self._adv_n_laps = n_laps
            self.adv_unwrapped_yaw = yaw_w + 2 * np.pi * n_laps
            self._adv_previous_yaw = yaw_w
        else:
            self._n_laps = n_laps
            self.unwrapped_yaw = yaw_w + 2 * np.pi * n_laps
            self._previous_yaw = yaw_w
            # Only ego updates both drones' previous actions
            self._previous_actions = self._actions.clone()
            self._adv_previous_actions = self._adv_actions.clone()

        return {obs_key: obs} if is_adversary else {"policy": obs}

    def _get_ego_observations(self) -> dict:
        """Get observations for ego drone."""
        return self._get_drone_observations(is_adversary=False)

    def _get_adversary_observations(self):
        """Get observations for adversary drone."""
        return self._get_drone_observations(is_adversary=True)

    #################################
    ### MULTI-AGENT METHODS BELOW ###
    #################################

    def _get_observations(self):
        """Return observations for both agents"""
        ego_obs = self._get_ego_observations()
        adv_obs = self._get_adversary_observations()

        return {
            "ego": ego_obs["policy"],
            "adversary": adv_obs["adv_policy"]
        }

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Return rewards for both agents directly"""
        ego_reward = self._get_ego_rewards()
        adv_reward = self._get_adv_rewards()

        return {
            "ego": ego_reward,
            "adversary": adv_reward
        }

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        terminated_dict, time_outs_dict, stuck_dict = self._get_dones_helper()

        # Store termination flags for use in rewards and logging
        self.reset_terminated = terminated_dict["ego"]
        self.adv_reset_terminated = terminated_dict["adversary"]
        self.reset_time_outs = time_outs_dict["ego"]
        self.adv_reset_time_outs = time_outs_dict["adversary"]

        # Store stuck status for passing to MAPPO
        self.ego_stuck = stuck_dict["ego"]
        self.adv_stuck = stuck_dict["adversary"]

        # Add stuck status to extras for passing through to MAPPO
        # This will be available in infos dict in record_transition
        self.extras["ego"]["stuck"] = stuck_dict["ego"]
        self.extras["adversary"]["stuck"] = stuck_dict["adversary"]

        # Track crash times for terminated episodes (not timeouts)
        ego_crashed = self.reset_terminated & ~self.reset_time_outs
        adv_crashed = self.adv_reset_terminated & ~self.adv_reset_time_outs

        self._ego_crash_times[ego_crashed] = self.episode_length_buf[ego_crashed].float() * self.step_dt
        self._adv_crash_times[adv_crashed] = self.episode_length_buf[adv_crashed].float() * self.step_dt

        return terminated_dict, time_outs_dict

    def _get_states(self) -> torch.Tensor:
        """Return global state for centralized critic"""
        ego_obs = self._get_ego_observations()["policy"]
        adv_obs = self._get_adversary_observations()["adv_policy"]

        # Combine observations for centralized critic
        return torch.cat([ego_obs, adv_obs], dim=-1)

    def reset(self, **kwargs):
        """Multi-agent reset"""
        super().reset()
        return self._get_observations(), {}



