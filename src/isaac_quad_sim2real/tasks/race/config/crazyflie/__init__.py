import gymnasium as gym

from . import agents, ma_quadcopter_env

##
# Register Gym environments.
##

# code for multi-agent environment registry
gym.register(
    id="Isaac-MA-Quadcopter-Race-v0",
    entry_point=ma_quadcopter_env.QuadcopterEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ma_quadcopter_env.QuadcopterEnvCfg,
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
