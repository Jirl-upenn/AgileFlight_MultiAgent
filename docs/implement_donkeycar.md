# Adapting AgileFlight Multi-Agent Racing to DonkeyCar + gym-donkeycar

This document provides a detailed architectural mapping between this repository's multi-agent drone racing stack and the DonkeyCar / gym-donkeycar ecosystem, along with a concrete implementation plan for adapting the competitive racing framework to ground vehicles.

---

## Table of Contents

1. [This Repository's Architecture in Detail](#1-this-repositorys-architecture-in-detail)
2. [DonkeyCar + gym-donkeycar Architecture in Detail](#2-donkeycar--gym-donkeycar-architecture-in-detail)
3. [Side-by-Side Mapping](#3-side-by-side-mapping)
4. [Implementation Plan](#4-implementation-plan)
5. [File-by-File Adaptation Guide](#5-file-by-file-adaptation-guide)
6. [Multi-Agent Considerations](#6-multi-agent-considerations)
7. [Runtime and Deployment Notes](#7-runtime-and-deployment-notes)

---

## 1. This Repository's Architecture in Detail

### 1.1 Directory Layout

```
src/isaac_quad_sim2real/
├── __init__.py                          # imports .tasks, triggering gym registration
├── tasks/
│   ├── __init__.py                      # auto-imports sub-packages via isaaclab_tasks.utils
│   └── race/
│       ├── __init__.py                  # "Drone racing environments." (docstring only)
│       └── config/
│           └── crazyflie/
│               ├── __init__.py          # gym.register("Isaac-MA-Quadcopter-Race-v0", ...)
│               ├── ma_quadcopter_env.py # QuadcopterEnvCfg + QuadcopterEnv (~1808 lines)
│               └── agents/
│                   ├── __init__.py
│                   ├── rl_cfg.py        # RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, etc.
│                   ├── rsl_rl_ppo_cfg.py
│                   └── skrl_mappo_cfg.yaml
scripts/skrl/
├── ma_train_race.py                     # Training entry point
├── ma_play_race.py                      # Evaluation entry point
└── cli_args.py                          # Shared CLI argument definitions
```

### 1.2 Environment Registration Flow

```
ma_train_race.py
  └── import src.isaac_quad_sim2real.tasks
        └── tasks/__init__.py calls import_packages()
              └── race/config/crazyflie/__init__.py
                    └── gym.register("Isaac-MA-Quadcopter-Race-v0",
                          entry_point=QuadcopterEnv,
                          kwargs={
                            "env_cfg_entry_point": QuadcopterEnvCfg,
                            "skrl_mappo_cfg_entry_point": "...agents:skrl_mappo_cfg.yaml"
                          })
```

The training script then calls `gym.make(args_cli.task, cfg=env_cfg)` which instantiates `QuadcopterEnv` with the merged config.

### 1.3 QuadcopterEnvCfg — Configuration Class

`QuadcopterEnvCfg(DirectMARLEnvCfg)` defines:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `possible_agents` | `["ego", "adversary"]` | Multi-agent identifiers |
| `action_spaces` | `{"ego": 4, "adversary": 4}` | Per-agent action dims |
| `observation_spaces` | `{"ego": 42 or 45, "adversary": 42 or 45}` | Obs dims (45 with walls, 42 without) |
| `state_space` | `84 or 90` | Centralized critic state (ego_obs + adv_obs) |
| `episode_length_s` | `30.0` | Max episode duration in seconds |
| `sim_rate_hz` | `500` | Physics simulation rate |
| `policy_rate_hz` | `50` | RL policy query rate |
| `decimation` | `10` | sim_rate / policy_rate |
| `pid_loop_rate_hz` | `500` | Inner PID control loop rate |
| `track` | `"complex"` or `"lemniscate"` | Track layout selection |
| `use_wall` | `bool` | Whether to spawn wall obstacles |
| `rewards` | `dict` | Reward scale factors (set by training script) |

Motor dynamics parameters (arm_length, k_eta, k_m, tau_m), PID gains (kp/ki/kd for roll/pitch/yaw), and domain randomization ranges are also defined here.

### 1.4 QuadcopterEnv — Environment Class

`QuadcopterEnv(DirectMARLEnv)` implements the full Isaac Lab MARL env interface:

#### Lifecycle Methods (called by DirectMARLEnv base class in this order):

1. **`_setup_scene()`** — Spawns two Crazyflie articulations (red ego, blue adversary), contact sensors, terrain, gates (from `usd/gate.usda`), walls, normal-vector arrows. Gates are USD references with PhysX collision (convex decomposition).

2. **`_pre_physics_step(actions: dict)`** — Receives `{"ego": Tensor[N,4], "adversary": Tensor[N,4]}`. Clamps to [-1,1], applies exponential smoothing (`beta`), converts thrust channel from [-1,1] to [0, max_force].

3. **`_apply_action()`** — Called every sim step (500 Hz). For each drone: runs PID body-rate controller → computes motor speeds → applies motor dynamics (first-order lag) → computes thrust/torque wrench → applies aerodynamic drag → calls `robot.set_external_force_and_torque()`.

4. **`_get_observations()`** — Returns `{"ego": Tensor[N,42/45], "adversary": Tensor[N,42/45]}`.

5. **`_get_rewards()`** — Returns `{"ego": Tensor[N], "adversary": Tensor[N]}`.

6. **`_get_dones()`** — Returns `(terminated_dict, timeout_dict)` per agent.

7. **`_get_states()`** — Returns `Tensor[N,84/90]` (concatenated ego+adv obs) for the centralized MAPPO critic.

8. **`_reset_idx(env_ids)`** — Resets specific environments: repositions drones behind random gates, randomizes PID gains and aerodynamic coefficients, resets counters.

#### Observation Vector (per drone, 42 or 45 dims):

| Component | Dims | Description |
|-----------|------|-------------|
| Global position (wall only) | 3 | World-frame position (only when `use_wall=True`) |
| Linear velocity | 3 | Body-frame velocity |
| Attitude matrix | 9 | Flattened 3×3 rotation matrix |
| Waypoint 1 vertices | 12 | 4 corners of current gate in body frame |
| Waypoint 2 vertices | 12 | 4 corners of next gate in body frame |
| Opponent position | 3 | Other drone's position in ego's body frame |
| Opponent velocity | 3 | Other drone's velocity in ego's body frame |

#### Action Vector (per drone, 4 dims):

| Channel | Range | Meaning |
|---------|-------|---------|
| `actions[:,0]` | [-1, 1] | Collective thrust (mapped to [0, weight × TWR]) |
| `actions[:,1]` | [-1, 1] | Roll rate command (scaled by `body_rate_scale_xy`) |
| `actions[:,2]` | [-1, 1] | Pitch rate command (scaled by `body_rate_scale_xy`) |
| `actions[:,3]` | [-1, 1] | Yaw rate command (scaled by `body_rate_scale_z`) |

#### Reward Function:

| Reward Component | Scale (default) | Trigger |
|-----------------|----------------|---------|
| `gate_pass` | `10.0` | +1.0 per gate passed (×0.5 if trailing) |
| `lap_bonus` | `0.5` (× 100.0 base) | Completing a full lap while leading |
| `roll_pitch_rates` | `-0.15` | Per-step penalty on roll/pitch rate magnitudes |
| `yaw_rate` | `-0.05` | Per-step penalty on yaw rate magnitude |
| `crash` | `-0.1` | Per-step cost while contact sensor is active |
| `death_cost` | `-2.0` | One-time penalty on terminal crash |

#### Termination Conditions:

- Gate collision (passed through gate plane but outside gate opening)
- Altitude out of bounds (`< 0.1m` after `max_time_on_ground`, or `> 3.0m`)
- Position out of flybox (track-specific X/Y boundaries)
- Flip (roll or pitch exceeding limits)
- Stuck for >100 timesteps (continuous contact sensor activation)
- Episode timeout (30s)

### 1.5 Training Pipeline (ma_train_race.py)

```python
# 1. Parse args (--task, --num_envs, --algorithm, --use_wall, --track, etc.)
# 2. Launch Isaac Sim via AppLauncher
# 3. Configure env_cfg (observation spaces, rewards, seed, etc.)
# 4. Create environment:
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=...)
# 5. Optionally convert MARL→single-agent:
if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
    env = multi_agent_to_single_agent(env)
# 6. Wrap for skrl:
env = SkrlVecEnvWrapper(env, ml_framework="torch")
# 7. Create and run skrl Runner:
runner = Runner(env, agent_cfg)  # agent_cfg from skrl_mappo_cfg.yaml
runner.run()
```

### 1.6 MAPPO Config (skrl_mappo_cfg.yaml)

```yaml
models:
  separate: True
  policy:  # GaussianMixin
    network: [{input: STATES, layers: [512, 512, 256, 128], activations: elu}]
  value:   # DeterministicMixin
    network: [{input: STATES, layers: [512, 512, 256, 256, 128, 128], activations: elu}]

agent:
  class: MAPPO
  rollouts: 16
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 1.0e-04
  grad_norm_clip: 0.5
  ratio_clip: 0.2
  value_clip: 0.2
  shared_state_preprocessor: RunningStandardScaler  # normalizes centralized state
```

---

## 2. DonkeyCar + gym-donkeycar Architecture in Detail

### 2.1 Project Relationship

- **[donkeycar](https://github.com/autorope/donkeycar)**: Full self-driving car platform (physical + sim). Uses a "parts" pipeline architecture.
- **[gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar)**: Gymnasium-compatible wrapper around the DonkeyCar Unity simulator. This is the primary integration point for RL.
- **[sdsandbox](https://github.com/tawnkramer/sdsandbox)**: The Unity simulator source (provides pre-built binaries).

### 2.2 gym-donkeycar Environment API

#### DonkeyEnv (gym.Env subclass)

```python
import gym
env = gym.make("donkey-minimonaco-track-v0", conf=conf)
obs, info = env.reset()      # obs: np.ndarray (120, 160, 3) uint8 RGB image
obs, reward, done, truncated, info = env.step([steering, throttle])
```

#### Observation Space

```python
spaces.Box(0, 255, shape=(120, 160, 3), dtype=np.uint8)
```

- Default: **120×160×3** RGB uint8 camera image from the front of the car
- Configurable via `cam_resolution` key in conf: `(height, width, channels)`
- Images arrive from Unity as base64-encoded JPG/PNG, decoded via PIL
- Optional second camera (`image_b`) for stereo
- Additional telemetry available via the handler (not in obs by default):
  - `pos_x, pos_y, pos_z` — world position
  - `speed` — forward speed
  - `cte` — cross-track error (distance from centerline)
  - `hit` — collision object name (`"none"` if no collision)
  - `gyro_x/y/z`, `accel_x/y/z`, `vel_x/y/z` — IMU-like data
  - `roll, pitch, yaw` — orientation
  - `lidar` — optional LIDAR point cloud

#### Action Space

```python
spaces.Box(
    low=np.array([-steer_limit, throttle_min]),
    high=np.array([steer_limit, throttle_max]),
    dtype=np.float32
)
# Default: steering [-1.0, 1.0], throttle [0.0, 1.0]
```

| Channel | Range | Meaning |
|---------|-------|---------|
| `action[0]` | [-1.0, 1.0] | Steering (left ↔ right) |
| `action[1]` | [0.0, 1.0] | Throttle (brake/stop ↔ full) |

#### Default Reward Function

```python
def calc_reward(done):
    if done:              return -1.0
    if cte > max_cte:     return -1.0
    if hit != "none":     return -2.0
    if forward_vel > 0:   return (1.0 - abs(cte) / max_cte) * forward_vel
    else:                 return forward_vel  # negative reward for reversing
```

Custom reward functions can be injected via `controller.set_reward_fn(fn)`.

#### Done Conditions

- Cross-track error exceeds `max_cte` (default 8.0)
- Collision with any object (`hit != "none"`)
- Missed checkpoint
- Disqualification
- Override: setting `RACE=True` env var disables auto-termination

#### Frame Skip

```python
# In DonkeyEnv.step():
for i in range(self.frame_skip):
    self.controller.take_action(action)
    # ... observe
```
Default `frame_skip=1`. Increasing this repeats the action for N simulator frames before returning, effectively lowering the policy rate.

### 2.3 Available Tracks (Gym IDs)

| Gym ID | Scene Name |
|--------|------------|
| `donkey-generated-roads-v0` | Generated Roads |
| `donkey-warehouse-v0` | Warehouse |
| `donkey-avc-sparkfun-v0` | AVC Sparkfun |
| `donkey-generated-track-v0` | Generated Track |
| `donkey-mountain-track-v0` | Mountain Track |
| `donkey-roboracingleague-track-v0` | Robo Racing League |
| `donkey-waveshare-v0` | Waveshare |
| `donkey-minimonaco-track-v0` | Mini Monaco |
| `donkey-warren-track-v0` | Warren Track |
| `donkey-thunderhill-track-v0` | Thunderhill |
| `donkey-circuit-launch-track-v0` | Circuit Launch |

### 2.4 Communication Architecture

```
┌─────────────┐    TCP/JSON     ┌──────────────────┐
│  Python      │◄──────────────►│  Unity Simulator  │
│  SDClient    │  port 9091     │  (sdsandbox)      │
│  (socket +   │                │                   │
│   bg thread) │                │  PhysX + Rendering│
└──────┬───────┘                └──────────────────┘
       │
┌──────▼───────┐
│  SimClient    │  Routes msg_type to handler
└──────┬───────┘
       │
┌──────▼───────────────────┐
│  DonkeyUnitySimHandler   │  Telemetry parsing, reward, done logic
└──────┬───────────────────┘
       │
┌──────▼───────────────────┐
│  DonkeyUnitySimController│  High-level API: take_action, observe, reset
└──────┬───────────────────┘
       │
┌──────▼───────┐
│  DonkeyEnv   │  gym.Env: step, reset, obs/action spaces
└──────────────┘
```

#### Key Message Types (Python → Unity)

| msg_type | Fields | Purpose |
|----------|--------|---------|
| `control` | `steering`, `throttle`, `brake` | Drive commands |
| `reset_car` | — | Reset car to start |
| `load_scene` | `scene_name` | Load a track |
| `cam_config` | `img_w/h/d`, `img_enc`, `fov`, `fish_eye_x/y`, `offset_x/y/z`, `rot_x/y/z` | Camera setup |
| `car_config` | `body_style`, `body_r/g/b`, `car_name` | Car appearance |

#### Key Message Types (Unity → Python)

| msg_type | Fields | Purpose |
|----------|--------|---------|
| `telemetry` | `image` (base64), `pos_x/y/z`, `speed`, `cte`, `hit`, `gyro_*`, `accel_*`, `vel_*`, `roll/pitch/yaw`, optional `lidar` | Per-frame state |
| `scene_selection_ready` | — | Simulator ready |
| `car_loaded` | — | Car spawned |
| `cross_start` | — | Crossed start line |
| `DQ` | — | Disqualification |

### 2.5 Configuration Dict

```python
conf = {
    # Simulator
    "exe_path": "/path/to/DonkeySimLinux.x86_64",  # or "remote"
    "host": "127.0.0.1",
    "port": 9091,
    "start_delay": 5.0,         # seconds to wait after launching Unity

    # Car appearance
    "body_style": "donkey",     # "donkey", "bare", "car01", "f1", "cybertruck"
    "body_rgb": (128, 128, 128),
    "car_name": "RL_agent",
    "font_size": 100,

    # Racer info
    "racer_name": "MAPPO",
    "country": "US",
    "bio": "Multi-agent competitive racing",
    "guid": "unique-string",

    # Environment
    "max_cte": 8.0,             # cross-track error limit
    "frame_skip": 1,            # action repeat
    "steer_limit": 1.0,        # max steering magnitude
    "throttle_min": 0.0,
    "throttle_max": 1.0,

    # Camera
    "cam_resolution": (120, 160, 3),  # (H, W, C)
    "cam_config": {
        "img_w": 160, "img_h": 120, "img_d": 3,
        "img_enc": "JPG",
        "fov": 90,
        "fish_eye_x": 0.0, "fish_eye_y": 0.0,
        "offset_x": 0.0, "offset_y": 0.0, "offset_z": 0.0,
        "rot_x": 0.0, "rot_y": 0.0, "rot_z": 0.0,
    },
}
```

### 2.6 donkeycar Parts Architecture

The donkeycar platform uses a **parts pipeline** where `Vehicle` orchestrates:

```python
class Vehicle:
    def __init__(self):
        self.mem = Memory()   # shared key-value store
        self.parts = []

    def add(self, part, inputs=[], outputs=[], threaded=False, run_condition=None):
        # Register part with its input/output memory keys

    def start(self, rate_hz=10):
        # Start threaded parts as daemon threads
        # Main loop: for each part, read inputs from Memory → call run() → write outputs to Memory
```

A `DonkeyGymEnv` part in `donkeycar/parts/dgym.py` wraps gym-donkeycar as a donkeycar part, bridging the two worlds.

---

## 3. Side-by-Side Mapping

### 3.1 Conceptual Mapping

| This Repo Concept | DonkeyCar Equivalent | Adaptation Notes |
|-------------------|---------------------|-----------------|
| `DirectMARLEnv` base class | `gym.Env` (DonkeyEnv) | DonkeyCar uses standard Gymnasium. No built-in MARL base class. |
| `QuadcopterEnvCfg` | `conf` dict | Replace `@configclass` with a plain Python config dict or dataclass |
| `_setup_scene()` — spawn robots, gates, terrain | `DonkeyUnityProcess.start()` + `load_scene` msg | Unity handles scene setup; Python only selects the track |
| `_pre_physics_step()` — action preprocessing | `DonkeyEnv.step()` → `controller.take_action()` | Simplify: 2D action, no PID, no motor model |
| `_apply_action()` — PID → motor → wrench | Unity physics handles all vehicle dynamics | **Eliminated entirely** — Unity's internal car controller takes steering+throttle directly |
| `_get_observations()` — state vectors | `controller.observe()` → camera image + telemetry | **Major change**: image-based obs (120×160×3) instead of state vector |
| `_get_rewards()` — gate pass + penalties | `handler.calc_reward()` | Rewrite: CTE-based + speed + lap rewards |
| `_get_dones()` — crash/flip/boundary checks | `handler.determine_episode_over()` | CTE limit, collision, missed checkpoint |
| `_get_states()` — centralized critic state | Concatenate both agents' obs | Must flatten/encode images or use separate state channels |
| `_reset_idx()` — position randomization | `controller.reset()` → sends `reset_car` msg | Unity handles repositioning; no manual pose control |
| `SkrlVecEnvWrapper` | Custom wrapper needed | See implementation plan |
| 4096 parallel envs on GPU | 1 env per Unity instance (CPU-bound) | **Critical limitation** — see parallelization strategy |

### 3.2 Observation Space Mapping

| Drone Obs Component | DonkeyCar Equivalent | Source |
|---------------------|---------------------|--------|
| Global position (3) | `pos_x, pos_y, pos_z` | Telemetry |
| Linear velocity (3) | `vel_x, vel_y, vel_z` or `speed` | Telemetry |
| Attitude matrix (9) | `roll, pitch, yaw` (3) | Telemetry (reduced) |
| Gate vertices (24) | Camera image (120×160×3) | Visual obs — the track IS the gates |
| Opponent position (3) | Not directly available | Must implement (see multi-agent section) |
| Opponent velocity (3) | Not directly available | Must implement (see multi-agent section) |

**Recommended DonkeyCar observation vector** (state-based mode):

| Component | Dims | Source |
|-----------|------|--------|
| Camera image | 120×160×3 (57,600) | Telemetry `image` field |
| Speed | 1 | Telemetry `speed` |
| CTE | 1 | Telemetry `cte` |
| Steering (previous) | 1 | Action history |
| Throttle (previous) | 1 | Action history |
| **Total (image mode)** | **57,604** | |

Or for a lower-dimensional state-based approach (no camera):

| Component | Dims | Source |
|-----------|------|--------|
| Position | 3 | `pos_x/y/z` |
| Velocity | 3 | `vel_x/y/z` |
| Orientation | 3 | `roll/pitch/yaw` |
| CTE | 1 | `cte` |
| Speed | 1 | `speed` |
| Previous action | 2 | Steering + throttle |
| **Total (state mode)** | **13** | |

### 3.3 Action Space Mapping

| Drone Action | DonkeyCar Action | Notes |
|-------------|-----------------|-------|
| Collective thrust [-1,1] | Throttle [0,1] | Different range; throttle is unsigned |
| Roll rate [-1,1] | Steering [-1,1] | Conceptually analogous (lateral control) |
| Pitch rate [-1,1] | *(no equivalent)* | Pitch doesn't exist for ground vehicles |
| Yaw rate [-1,1] | *(implicit via steering)* | Yaw is a consequence of steering, not direct |

### 3.4 Reward Mapping

| Drone Reward | DonkeyCar Equivalent | Implementation |
|-------------|---------------------|----------------|
| `gate_pass` (+10.0) | Checkpoint/lap progress | Track `cross_start` and checkpoint events |
| `lap_bonus` (+50.0) | Lap completion bonus | Detect `cross_start` after full lap |
| `roll_pitch_rates` (-0.15) | Steering smoothness penalty | `-scale * (steering_delta²)` |
| `yaw_rate` (-0.05) | *(fold into steering smoothness)* | |
| `crash` (-0.1/step) | Collision penalty | `-2.0` per collision (from `hit != "none"`) |
| `death_cost` (-2.0) | Off-track penalty | `-1.0` when `cte > max_cte` |
| *(none)* | **CTE reward** (new) | `+(1 - abs(cte)/max_cte) * speed` |
| *(none)* | **Speed reward** (new) | `+speed_scale * forward_velocity` |

---

## 4. Implementation Plan

### Phase 1: Single-Agent DonkeyCar Environment

#### Step 1.1: Install Dependencies

```bash
# Install gym-donkeycar
pip install gym-donkeycar

# Or from source for latest:
git clone https://github.com/tawnkramer/gym-donkeycar.git
cd gym-donkeycar && pip install -e .

# Download Unity simulator binary from:
# https://github.com/tawnkramer/gym-donkeycar/releases
# Extract to a known path, e.g., ~/DonkeySimLinux/donkey_sim.x86_64
chmod +x ~/DonkeySimLinux/donkey_sim.x86_64
```

#### Step 1.2: Create Task Directory

```
src/isaac_quad_sim2real/tasks/
├── race/          # existing drone racing
└── donkeycar/     # new
    ├── __init__.py
    └── config/
        ├── __init__.py
        ├── donkey_env.py          # DonkeyRaceEnv + DonkeyRaceEnvCfg
        └── agents/
            ├── __init__.py
            └── skrl_ppo_cfg.yaml  # PPO config for single-agent
```

#### Step 1.3: Implement DonkeyRaceEnv

This is the core adaptation. Since DonkeyCar uses its own Unity simulator (not Isaac Sim), the environment **cannot** inherit from `DirectMARLEnv` or `DirectRLEnv`. Instead, create a standalone `gym.Env` subclass that presents the same interface the skrl Runner expects after wrapping.

```python
# src/isaac_quad_sim2real/tasks/donkeycar/config/donkey_env.py

import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
import gym_donkeycar  # registers donkey-* environments


@dataclass
class DonkeyRaceEnvCfg:
    """Configuration for DonkeyCar racing environment."""
    # Simulator
    exe_path: str = "remote"           # path to Unity binary, or "remote"
    host: str = "127.0.0.1"
    port: int = 9091
    start_delay: float = 5.0

    # Environment
    donkey_env_id: str = "donkey-minimonaco-track-v0"
    max_cte: float = 8.0
    frame_skip: int = 2                # ~15 Hz effective policy rate at 30 Hz sim
    episode_length_s: float = 60.0

    # Observation mode
    obs_mode: str = "state"            # "image" or "state"
    cam_resolution: tuple = (120, 160, 3)
    use_grayscale: bool = False
    frame_stack: int = 1               # number of frames to stack (for image mode)

    # Action
    steer_limit: float = 1.0
    throttle_min: float = 0.0
    throttle_max: float = 1.0

    # Rewards
    rewards: dict = field(default_factory=lambda: {
        'cte_reward_scale': 1.0,       # centerline tracking
        'speed_reward_scale': 0.5,     # forward progress
        'lap_bonus_reward_scale': 10.0,
        'steering_smoothness_scale': -0.1,
        'collision_penalty': -2.0,
        'off_track_penalty': -1.0,
    })

    # Multi-agent (Phase 2)
    possible_agents: list = field(default_factory=lambda: ["ego"])
    num_agents: int = 1

    # Car appearance
    body_style: str = "donkey"
    body_rgb: tuple = (128, 128, 128)
    car_name: str = "RL_agent"


class DonkeyRaceEnv(gym.Env):
    """DonkeyCar racing environment adapted for the AgileFlight training stack.

    Wraps gym-donkeycar to present an interface compatible with the skrl
    training pipeline used by this repository.
    """

    def __init__(self, cfg: DonkeyRaceEnvCfg):
        super().__init__()
        self.cfg = cfg
        self.device = "cpu"  # DonkeyCar runs on CPU (Unity sim)

        # Build conf dict for gym-donkeycar
        self._conf = {
            "exe_path": cfg.exe_path,
            "host": cfg.host,
            "port": cfg.port,
            "start_delay": cfg.start_delay,
            "max_cte": cfg.max_cte,
            "frame_skip": cfg.frame_skip,
            "steer_limit": cfg.steer_limit,
            "throttle_min": cfg.throttle_min,
            "throttle_max": cfg.throttle_max,
            "cam_resolution": cfg.cam_resolution,
            "body_style": cfg.body_style,
            "body_rgb": cfg.body_rgb,
            "car_name": cfg.car_name,
            "guid": str(id(self)),
        }

        # Create underlying donkey env
        self._env = gym.make(cfg.donkey_env_id, conf=self._conf)

        # Define spaces based on obs_mode
        if cfg.obs_mode == "image":
            h, w, c = cfg.cam_resolution
            if cfg.use_grayscale:
                c = 1
            c *= cfg.frame_stack
            self.observation_space = gym.spaces.Box(0, 255, (h, w, c), dtype=np.uint8)
        else:
            # State-based: [pos(3), vel(3), orient(3), cte(1), speed(1), prev_action(2)]
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=np.array([-cfg.steer_limit, cfg.throttle_min]),
            high=np.array([cfg.steer_limit, cfg.throttle_max]),
            dtype=np.float32,
        )

        # State tracking
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_steering = 0.0
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._frame_buffer = []

        # Episode logging (mirrors drone env's extras["log"] pattern)
        self._episode_sums = {k.split("_scale")[0]: 0.0 for k in cfg.rewards}
        self.extras = {"log": {}}

    def _build_state_obs(self, info: dict) -> np.ndarray:
        """Build state-based observation from telemetry."""
        return np.array([
            info.get("pos_x", 0.0),
            info.get("pos_y", 0.0),
            info.get("pos_z", 0.0),
            info.get("vel_x", 0.0),
            info.get("vel_y", 0.0),
            info.get("vel_z", 0.0),
            info.get("roll", 0.0),
            info.get("pitch", 0.0),
            info.get("yaw", 0.0),
            info.get("cte", 0.0),
            info.get("speed", 0.0),
            self._prev_action[0],
            self._prev_action[1],
        ], dtype=np.float32)

    def _compute_reward(self, obs, done, info) -> float:
        """Compute shaped reward analogous to the drone racing reward structure."""
        rew = self.cfg.rewards
        reward = 0.0

        cte = info.get("cte", 0.0)
        speed = info.get("speed", 0.0)
        hit = info.get("hit", "none")

        # CTE reward (track centering)
        if self.cfg.max_cte > 0:
            cte_reward = (1.0 - abs(cte) / self.cfg.max_cte) * max(speed, 0.0)
            reward += rew['cte_reward_scale'] * cte_reward

        # Speed reward
        reward += rew['speed_reward_scale'] * max(speed, 0.0)

        # Steering smoothness (analogous to roll_pitch_rates penalty)
        steering_delta = self._prev_action[0] - self._prev_steering
        reward += rew['steering_smoothness_scale'] * (steering_delta ** 2)

        # Collision penalty
        if hit != "none":
            reward += rew['collision_penalty']

        # Off-track penalty (terminal)
        if done and abs(cte) > self.cfg.max_cte:
            reward += rew['off_track_penalty']

        return reward

    def step(self, action):
        self._prev_steering = self._prev_action[0]
        obs, _, done, truncated, info = self._env.step(action)
        self._prev_action = np.array(action, dtype=np.float32)
        self._episode_steps += 1

        # Check episode time limit
        max_steps = int(self.cfg.episode_length_s * 30 / max(self.cfg.frame_skip, 1))
        if self._episode_steps >= max_steps:
            truncated = True

        reward = self._compute_reward(obs, done, info)
        self._episode_reward += reward

        if self.cfg.obs_mode == "state":
            obs = self._build_state_obs(info)

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        # Log episode stats before reset (mirrors drone env pattern)
        if self._episode_steps > 0:
            self.extras["log"] = {
                "Episode_Reward/total": self._episode_reward,
                "Episode_Length/steps": self._episode_steps,
            }

        obs, info = self._env.reset(**kwargs)
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_steering = 0.0
        self._episode_steps = 0
        self._episode_reward = 0.0

        if self.cfg.obs_mode == "state":
            obs = self._build_state_obs(info)

        return obs, info

    def close(self):
        self._env.close()
```

#### Step 1.4: Register the Environment

```python
# src/isaac_quad_sim2real/tasks/donkeycar/config/__init__.py

import gymnasium as gym
from . import donkey_env
from . import agents

gym.register(
    id="Donkeycar-Race-v0",
    entry_point=donkey_env.DonkeyRaceEnv,
    disable_env_checker=True,
    kwargs={
        "cfg": donkey_env.DonkeyRaceEnvCfg(),
    },
)
```

#### Step 1.5: Create Training Script

```python
# scripts/skrl/donkey_train.py
# Standalone script — does NOT use Isaac Sim AppLauncher

import argparse
import gymnasium as gym
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/third_parties"))

from skrl.utils.runner.torch import Runner

import src.isaac_quad_sim2real.tasks  # triggers gym registration

parser = argparse.ArgumentParser()
parser.add_argument("--exe_path", type=str, default="remote")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=9091)
parser.add_argument("--track", type=str, default="donkey-minimonaco-track-v0")
parser.add_argument("--obs_mode", type=str, default="state", choices=["state", "image"])
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--frame_skip", type=int, default=2)
args = parser.parse_args()

# Build config
from src.isaac_quad_sim2real.tasks.donkeycar.config.donkey_env import DonkeyRaceEnvCfg
cfg = DonkeyRaceEnvCfg(
    exe_path=args.exe_path,
    host=args.host,
    port=args.port,
    donkey_env_id=args.track,
    obs_mode=args.obs_mode,
    frame_skip=args.frame_skip,
)

env = gym.make("Donkeycar-Race-v0", cfg=cfg)

# Load YAML config and create runner
# ... (load skrl_ppo_cfg.yaml, configure runner, run)
```

### Phase 2: Multi-Agent DonkeyCar Racing

#### Multi-Agent Architecture

The sdsandbox Unity simulator supports **multiple cars via multiple TCP connections**. Each connected client controls one car and sees all other cars in the scene. This enables competitive multi-agent racing.

```
┌──────────┐  port 9091   ┌─────────────────────┐
│ Agent 0  │◄────────────►│                     │
│ (ego)    │              │  Unity Simulator     │
└──────────┘              │  (sdsandbox)         │
                          │                     │
┌──────────┐  port 9091   │  Both cars visible   │
│ Agent 1  │◄────────────►│  to each other       │
│ (adv)    │              │                     │
└──────────┘              └─────────────────────┘
```

#### DonkeyMARaceEnv (Multi-Agent Wrapper)

```python
class DonkeyMARaceEnv:
    """Multi-agent DonkeyCar racing environment.

    Manages two DonkeyEnv instances connected to the same Unity simulator,
    presenting a dict-based multi-agent interface compatible with MAPPO.
    """
    possible_agents = ["ego", "adversary"]

    def __init__(self, cfg: DonkeyRaceEnvCfg):
        # Create two gym-donkeycar envs on the SAME host:port
        # Each connects as a separate TCP client
        self._envs = {}
        for i, agent in enumerate(self.possible_agents):
            agent_conf = {**self._base_conf}
            agent_conf["car_name"] = agent
            agent_conf["body_rgb"] = (255, 0, 0) if agent == "ego" else (0, 0, 255)
            agent_conf["guid"] = f"{agent}_{id(self)}"
            # Each agent needs its own connection
            # The simulator accepts multiple clients on the same port
            self._envs[agent] = gym.make(cfg.donkey_env_id, conf=agent_conf)

    def step(self, actions: dict) -> tuple:
        # Step both agents
        observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}
        for agent in self.possible_agents:
            obs, rew, term, trunc, info = self._envs[agent].step(actions[agent])
            observations[agent] = obs
            rewards[agent] = rew
            terminateds[agent] = term
            truncateds[agent] = trunc
            infos[agent] = info
        return observations, rewards, terminateds, truncateds, infos
```

**Important**: The sdsandbox simulator handles multi-car by accepting one car per TCP client. Both clients connect to the same `host:port`. Each sees the other car rendered in the scene. Telemetry for the OTHER car is not directly available in the standard protocol — you would need to add opponent position to the observation by:

1. **Option A**: Modify sdsandbox Unity code to include opponent telemetry in each client's telemetry message
2. **Option B**: Share telemetry between the two Python-side handlers via shared memory / a coordinator class
3. **Option C**: Rely purely on camera images (the opponent car IS visible in the camera feed)

#### Centralized State for MAPPO

For `_get_states()` (centralized critic), concatenate both agents' observations plus cross-agent telemetry:

```python
def _get_states(self):
    ego_tel = self._envs["ego"].handler.telemetry
    adv_tel = self._envs["adversary"].handler.telemetry
    return np.concatenate([
        ego_state_vector,    # 13 dims
        adv_state_vector,    # 13 dims
        relative_position,   # 3 dims (ego→adv in ego's frame)
        relative_velocity,   # 3 dims
    ])  # Total: 32 dims
```

---

## 5. File-by-File Adaptation Guide

| This Repo File | DonkeyCar Equivalent | Key Changes |
|---------------|---------------------|-------------|
| `ma_quadcopter_env.py:QuadcopterEnvCfg` | `donkey_env.py:DonkeyRaceEnvCfg` | Replace Isaac Lab cfg with plain dataclass. Drop motor/PID/gate params. Add cam_resolution, exe_path, max_cte. |
| `ma_quadcopter_env.py:QuadcopterEnv.__init__()` | `DonkeyRaceEnv.__init__()` | Drop Isaac Lab scene setup. Create gym-donkeycar env instead. |
| `ma_quadcopter_env.py:_setup_scene()` | *(eliminated)* | Unity handles scene. Just call `load_scene` via gym-donkeycar. |
| `ma_quadcopter_env.py:_pre_physics_step()` | *(eliminated)* | No action preprocessing needed (no PID, no motor model). |
| `ma_quadcopter_env.py:_apply_action()` | `DonkeyEnv.step()` calls `take_action()` | Unity handles all vehicle physics. |
| `ma_quadcopter_env.py:_get_observations()` | `_build_state_obs()` or raw camera image | 13-dim state vector or 120×160×3 image |
| `ma_quadcopter_env.py:_calculate_drone_rewards()` | `_compute_reward()` | CTE-based instead of gate-based. Keep reward logging structure. |
| `ma_quadcopter_env.py:_get_dones_helper()` | gym-donkeycar's built-in `done` + custom timeout | CTE limit + collision instead of flybox + flip |
| `ma_quadcopter_env.py:_reset_idx()` | `DonkeyEnv.reset()` → sends `reset_car` | No manual position randomization (Unity handles it). |
| `ma_quadcopter_env.py:_get_states()` | Concatenate agent obs | Same pattern, smaller dims |
| `skrl_mappo_cfg.yaml` | `skrl_ppo_cfg.yaml` (single) or `skrl_mappo_cfg.yaml` (multi) | Smaller networks. Change `input: STATES` dims. For image mode, add CNN encoder. |
| `ma_train_race.py` | `donkey_train.py` | Remove Isaac Sim AppLauncher. Remove `use_wall`/`track` args. Add `--exe_path`, `--host`, `--port`, `--track` (donkey env id). |
| `ma_play_race.py` | `donkey_play.py` | Same simplification as training script. |

---

## 6. Multi-Agent Considerations

### 6.1 What Works Out of the Box

- The sdsandbox Unity simulator accepts multiple TCP clients simultaneously
- Each client controls one car, and all cars are rendered in the shared scene
- Cars can collide with each other in the Unity physics engine
- Camera observations naturally include other cars (visual multi-agent)

### 6.2 What Requires Work

| Feature | Status | Work Needed |
|---------|--------|------------|
| Multiple car spawning | Built-in | Each `gym.make()` call connects a new car |
| Opponent in camera | Built-in | Car is visible in RGB observation |
| Opponent state vector | Not available | Share telemetry between Python handlers (Option B above) |
| Collision between cars | Built-in (Unity PhysX) | `hit` field may report opponent name |
| Synchronized stepping | Not built-in | Need a coordinator to sync `step()` calls |
| Centralized state | Not built-in | Aggregate both agents' telemetry |
| Competitive rewards (leader-aware) | Not built-in | Track lap progress per agent, adapt drone's leader-shaping logic |

### 6.3 Synchronization Strategy

The drone env steps all environments in lockstep on GPU. DonkeyCar uses async TCP messaging. To synchronize:

```python
class SyncedMultiAgentStep:
    """Ensures both agents step before either observes."""
    def step(self, actions):
        # 1. Send both actions (non-blocking)
        for agent in self.possible_agents:
            self._envs[agent].controller.take_action(actions[agent])

        # 2. Wait for both telemetry responses
        for agent in self.possible_agents:
            self._observations[agent] = self._envs[agent].controller.observe()

        # 3. Return synchronized results
        return self._observations, self._rewards, ...
```

### 6.4 Leader-Aware Reward Adaptation

The drone env's leader-detection logic transfers directly:

```python
# From drone env — adapt for DonkeyCar:
# Instead of gate-passing count, use lap progress (distance along track centerline)
ego_progress = ego_telemetry["lap_progress"]  # needs custom Unity metric
adv_progress = adv_telemetry["lap_progress"]
is_leading = ego_progress > adv_progress

# Gate pass bonus → checkpoint/lap bonus
# Leading agent gets full bonus, trailing gets half
checkpoint_bonus = base_bonus * (1.0 if is_leading else 0.5)
```

---

## 7. Runtime and Deployment Notes

### 7.1 Performance: Isaac Lab vs DonkeyCar

| Aspect | Drone (Isaac Lab) | DonkeyCar |
|--------|-------------------|-----------|
| Parallel envs | 4096+ on GPU | 1 per Unity instance |
| Steps/second | ~50,000 (vectorized) | ~30 per instance |
| Rendering | Optional (headless) | Required for camera obs, optional for state obs |
| GPU usage | Physics + policy | Rendering only (if camera) + policy |

**Parallelization strategies for DonkeyCar:**
1. **Multiple Unity instances**: Launch N simulator binaries on ports 9091..9091+N, each running one (or two for multi-agent) cars. Use `gym.vector.AsyncVectorEnv` or `SubprocVecEnv`.
2. **Headless mode**: Pass `--headless` to the Unity binary to skip rendering (only valid for state-based obs, not camera).
3. **Reduced resolution**: Use 64×64 grayscale instead of 120×160 RGB to reduce image transfer overhead.

### 7.2 Network Sizing

For **state-based** (13-dim) observations:
```yaml
policy:
  network: [{layers: [128, 128, 64], activations: elu}]
value:
  network: [{layers: [128, 128, 64], activations: elu}]
```

For **image-based** (120×160×3) observations, add a CNN encoder:
```yaml
policy:
  network:
    - name: cnn
      input: STATES
      layers: [Conv2d(3,32,8,4), Conv2d(32,64,4,2), Conv2d(64,64,3,1), Flatten]
      activations: relu
    - name: mlp
      input: cnn
      layers: [256, 128]
      activations: elu
  output: ACTIONS
```

### 7.3 Control Frequency Alignment

| Parameter | Drone Env | DonkeyCar (Recommended) |
|-----------|-----------|------------------------|
| Sim rate | 500 Hz | ~30 Hz (Unity fixed) |
| Policy rate | 50 Hz | 15 Hz (frame_skip=2) |
| Decimation | 10 | 2 |

Set `frame_skip=2` in the DonkeyCar config to get ~15 Hz policy rate, which is reasonable for ground vehicle control.

### 7.4 Quick Start Commands

```bash
# Single-agent training (state-based, connect to running simulator)
python scripts/skrl/donkey_train.py \
    --exe_path remote \
    --host 127.0.0.1 \
    --port 9091 \
    --track donkey-minimonaco-track-v0 \
    --obs_mode state \
    --max_iterations 5000

# Single-agent training (launch simulator automatically)
python scripts/skrl/donkey_train.py \
    --exe_path ~/DonkeySimLinux/donkey_sim.x86_64 \
    --track donkey-warehouse-v0 \
    --obs_mode image \
    --frame_skip 2

# Multi-agent training (Phase 2)
python scripts/skrl/donkey_ma_train.py \
    --exe_path ~/DonkeySimLinux/donkey_sim.x86_64 \
    --track donkey-minimonaco-track-v0 \
    --algorithm MAPPO \
    --obs_mode state \
    --max_iterations 10000
```

### 7.5 Validation Checklist

1. **Smoke test**: `env.reset()` + 100 random actions without Python crash
2. **Observation sanity**: Verify image shape matches `cam_resolution`; verify state vector has correct telemetry values
3. **Reward sanity**: Train PPO for 1000 steps — reward should increase and car should learn to not immediately drive off-track
4. **Logging**: Verify `extras["log"]` contains episode reward/length data compatible with wandb logging
5. **Multi-agent sync**: Both cars visible in each other's camera; `step()` returns within bounded time
6. **Checkpoint save/load**: Save a checkpoint and reload it in `donkey_play.py` to verify inference works
