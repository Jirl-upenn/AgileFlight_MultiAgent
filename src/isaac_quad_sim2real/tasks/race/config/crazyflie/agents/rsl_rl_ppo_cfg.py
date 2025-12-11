# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticRecurrentCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 200
    save_interval = 50
    experiment_name = "quadcopter_direct"
    empirical_normalization = False
    wandb_project = "single-agent-sim2real"
    wandb_entity = "vineetp-university-of-pennsylvania"

    # ego drone: recurrent FiLM architecture
    # policy = RslRlPpoActorCriticRecurrentCfg(
    #     class_name="ActorCriticRecurrentFiLM",
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[128, 128],
    #     film_hidden_dims=[3, 3],
    #     cond_dim=2,
    #     critic_hidden_dims=[512, 512],
    #     activation="elu",
    #     rnn_type="lstm",
    #     rnn_hidden_size=256,
    #     rnn_num_layers=2,
    #     min_std=0.2,
    # )
    
    # ego drone: non-recurrent FiLM architecture (for compatibility with saved checkpoint)
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        film_hidden_dims=[3, 3],
        cond_dim=2,
        critic_hidden_dims=[512, 512],
        activation="elu",
        min_std=0.2,
    )

    # adversary drone: non-recurrent FiLM
    adversary_policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128],
        film_hidden_dims=[3, 3],
        cond_dim=2,
        critic_hidden_dims=[512, 512],
        activation="elu",
        min_std=0.2,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
