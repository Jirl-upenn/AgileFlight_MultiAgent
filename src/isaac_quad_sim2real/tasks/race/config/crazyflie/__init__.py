import gymnasium as gym

from . import agents, quadcopter_env

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Quadcopter-Race-v0",
    entry_point=quadcopter_env.QuadcopterEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadcopter_env.QuadcopterEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)
