from agents.actor_critic_agents.A2C import A2C
from agents.actor_critic_agents.A3C import A3C
from agents.actor_critic_agents.DDPG_HER import DDPG_HER
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.TD3 import TD3
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.DDQN_With_Prioritised_Experience_Replay import DDQN_With_Prioritised_Experience_Replay
from agents.DQN_agents.DQN_HER import DQN_HER
from agents.DQN_agents.DQN_With_Fixed_Q_Targets import DQN_With_Fixed_Q_Targets
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from agents.DQN_agents.DDQN import DDQN
from agents.hierarchical_agents.DIAYN import DIAYN
from agents.hierarchical_agents.h_DQN import h_DQN
from agents.hierarchical_agents.HIRO import HIRO
from agents.hierarchical_agents.SNN_HRL import SNN_HRL
from agents.policy_gradient_agents.PPO import PPO
from agents.policy_gradient_agents.REINFORCE import REINFORCE

from environments.FaceDiscreete import FaceEnvironementDiscreete
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

config = Config()
config.seed = 1

config.environment = FaceEnvironementDiscreete(
    "../weights/blg_small_12_5e-06_5e-05_2_8_small_big_noisy_first_True_512")

config.num_episodes_to_run = 500
config.file_to_save_data_results = "Data_and_Graphs/FaceDiscreete.pkl"
config.file_to_save_results_graph = "Data_and_Graphs/FaceDiscreete.png"
config.show_solution_score = True
config.visualise_individual_results = True
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = True
config.randomise_random_seed = True
config.save_model = True

actor_critic_agent_hyperparameters = {
    "Actor": {
        "learning_rate": 0.0003,
        "linear_hidden_units": [64, 64],
        "final_layer_activation": None,
        "batch_norm": False,
        "tau": 0.005,
        "gradient_clipping_norm": 5,
        "initialiser": "Xavier"
    },

    "Critic": {
        "learning_rate": 0.0003,
        "linear_hidden_units": [64, 64],
        "final_layer_activation": None,
        "batch_norm": False,
        "buffer_size": 1000000,
        "tau": 0.005,
        "gradient_clipping_norm": 5,
        "initialiser": "Xavier"
    },
    "HER_sample_proportion": 0.8,
    "min_steps_before_learning": 400,
    "batch_size": 256,
    "discount_rate": 0.99,
    "mu": 0.0,  # for O-H noise
    "theta": 0.15,  # for O-H noise
    "sigma": 0.25,  # for O-H noise
    "action_noise_std": 0.2,  # for TD3
    "action_noise_clipping_range": 0.5,  # for TD3
    "update_every_n_steps": 1,
    "learning_updates_per_learning_session": 1,
    "automatically_tune_entropy_hyperparameter": True,
    "entropy_term_weight": None,
    "add_extra_noise": False,
    "do_evaluation_iterations": True,
    "clip_rewards": False
}

dqn_agent_hyperparameters = {
    "learning_rate": 0.005,
    "batch_size": 128,
    "buffer_size": 40000,
    "epsilon": 1.0,
    "epsilon_decay_rate_denominator": 3,
    "discount_rate": 0.99,
    "tau": 0.01,
    "alpha_prioritised_replay": 0.6,
    "beta_prioritised_replay": 0.1,
    "incremental_td_error": 1e-8,
    "update_every_n_steps": 3,
    "linear_hidden_units": [30, 15],
    "final_layer_activation": "None",
    "batch_norm": False,
    "gradient_clipping_norm": 5,
    "clip_rewards": False
}


manager_hyperparameters = dqn_agent_hyperparameters
manager_hyperparameters.update({"timesteps_to_give_up_control_for": 5})


config.hyperparameters = {
    "HRL": {
        "linear_hidden_units": [32, 32],
        "learning_rate": 0.005,
        "buffer_size": 1000000,
        "batch_size":  256,
        "final_layer_activation": "None",
        # "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": False,
        "gradient_clipping_norm": 0.5,
        "update_every_n_steps": 1,
        "epsilon_decay_rate_denominator": 2,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau":  0.004,
        "sequitur_k": 2,
        "use_relative_counts": True,
        "action_length_reward_bonus": 0.0,
        "pre_training_learning_iterations_multiplier": 0,
        "episodes_to_run_with_no_exploration": 0,
        "action_balanced_replay_buffer": True,
        "copy_over_hidden_layers": True,
        "random_episodes_to_run": 0,
        "only_train_new_actions": True,
        "only_train_final_layer": True,
        "num_top_results_to_use": 10,
        "action_frequency_required_in_top_results": 0.8,
        "reduce_macro_action_appearance_cutoff_throughout_training": False,
        "add_1_macro_action_at_a_time": True,
        "calculate_q_values_as_increments": True,
        "episodes_per_round": 50,
        "abandon_ship": True,
        "clip_rewards": True
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.05,
        "linear_hidden_units": [30, 15],
        "final_layer_activation": "TANH",
        "learning_iterations_per_round": 10,
        "discount_rate": 0.9,
        "batch_norm": False,
        "clip_epsilon": 0.2,
        "episodes_per_learning_round": 10,
        "normalise_rewards": True,
        "gradient_clipping_norm": 5,
        "mu": 0.0,
        "theta": 0.15,
        "sigma": 0.2,
        "epsilon_decay_rate_denominator": 1,
        "clip_rewards": False
    },
    "DQN_Agents": {
        "linear_hidden_units": [32, 10],
        "learning_rate": 0.005,
        "buffer_size": 1000000,
        "batch_size": 256,
        "final_layer_activation": "None",
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.1,
        "epsilon": 1.0,
        "epsilon_decay_rate_denominator": 3,
        "incremental_td_error": 1e-8,
        # "columns_of_data_to_be_embedded": [0],
        # "embedding_dimensions": [[config.environment.observation_space.n, embedding_dimensionality]],
        "batch_norm": False,
        "gradient_clipping_norm":  0.5,
        "update_every_n_steps": 1,
        "discount_rate": 0.99,
        "learning_iterations": 1,
        "tau": 0.004,
        "clip_rewards": True,
        "HER_sample_proportion": 0.8
    },

    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [20, 10],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": True,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,
        "HER_sample_proportion": 0.8,
        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    },
    "DIAYN": {
        "DISCRIMINATOR": {
            "learning_rate": 0.001,
            "linear_hidden_units": [32, 32],
            "final_layer_activation": None,
            "gradient_clipping_norm": 5

        },
        "AGENT": actor_critic_agent_hyperparameters,
        "MANAGER": manager_hyperparameters,
        "num_skills": 10,
        "num_unsupservised_episodes": 500
    },
    "h_DQN": {
        "CONTROLLER": {
            "batch_size": 256,
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))],
                                     [config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1500,
            "discount_rate": 0.999,
            "learning_iterations": 1
        },
        "META_CONTROLLER": {
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 2500,
            "discount_rate": 0.999,
            "learning_iterations": 1
        }
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 2,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.99,
            "episodes_for_pretraining": 2000,
            # "batch_size": 256,
            # "learning_rate": 0.01,
            # "buffer_size": 40000,
            # "linear_hidden_units": [20, 10],
            # "final_layer_activation": "None",
            # "columns_of_data_to_be_embedded": [0, 1],
            # "embedding_dimensions": [[config.environment.observation_space.n,
            #                           max(4, int(config.environment.observation_space.n / 10.0))],
            #                          [6, 4]],
            # "batch_norm": False,
            # "gradient_clipping_norm": 5,
            # "update_every_n_steps": 1,
            # "epsilon_decay_rate_denominator": 50,
            # "discount_rate": 0.999,
            # "learning_iterations": 1


            "learning_rate": 0.05,
            "linear_hidden_units": [20, 20],
            "final_layer_activation": "SOFTMAX",
            "learning_iterations_per_round": 5,
            "discount_rate": 0.99,
            "batch_norm": False,
            "clip_epsilon": 0.1,
            "episodes_per_learning_round": 4,
            "normalise_rewards": True,
            "gradient_clipping_norm": 7.0,
            "mu": 0.0,  # only required for continuous action games
            "theta": 0.0,  # only required for continuous action games
            "sigma": 0.0,  # only required for continuous action games
            "epsilon_decay_rate_denominator": 1.0



        },

        "MANAGER": {
            "timesteps_before_changing_skill": 4,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 256,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[config.environment.observation_space.n,
                                      max(4, int(config.environment.observation_space.n / 10.0))]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 1000,
            "discount_rate": 0.999,
            "learning_iterations": 1

        }

    },
    "HIRO": {

        "LOWER_LEVEL": {
            "max_lower_level_timesteps": 5,

            "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "Critic": {
                "learning_rate": 0.01,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "None",
                "batch_norm": False,
                "buffer_size": 100000,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "batch_size": 256,
            "discount_rate": 0.9,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,
            "clip_rewards": False,
            "number_goal_candidates": 8

        },



        "HIGHER_LEVEL": {

            "Actor": {
                "learning_rate": 0.001,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "TANH",
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "number_goal_candidates": 8,
            },

            "Critic": {
                "learning_rate": 0.01,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": "None",
                "batch_norm": False,
                "buffer_size": 100000,
                "tau": 0.005,
                "gradient_clipping_norm": 5
            },

            "batch_size": 256,
            "discount_rate": 0.9,
            "mu": 0.0,  # for O-H noise
            "theta": 0.15,  # for O-H noise
            "sigma": 0.25,  # for O-H noise
            "action_noise_std": 0.2,  # for TD3
            "action_noise_clipping_range": 0.5,  # for TD3
            "update_every_n_steps": 20,
            "learning_updates_per_learning_session": 10,
            "number_goal_candidates": 8,
            "clip_rewards": False

        },


    },
}

if __name__ == '__main__':

    # AGENTS = [A2C, A3C, DDPG_HER, DDPG, SAC, SAC_Discrete, TD3, DDQN,
    #           DDQN_With_Prioritised_Experience_Replay, DQN_HER,
    #           DQN_With_Fixed_Q_Targets, Dueling_DDQN, DDQN, DIAYN, h_DQN,
    #           HIRO, SNN_HRL, PPO, REINFORCE]
    AGENTS = [DDQN]

# OK
# DDQN, DDQN_With_Prioritised_Experience_Replay, DQN_With_Fixed_Q_Targets,
# Dueling_DDQN, SAC_Discrete

# DDPG HIRO TD3 Multi action ?

# A2C et A3C probleme cuda multiprocess

# DIYAN SAC continu

# Her needs observation and changing goal

# SNN_HRL Only works for discrete states (no ldmk like us => discretise states)

# H_dqn AttributeError: 'DDQN' object has no attribute 'state'

# PPO Problem with multiprocessing

# REINFORCE RuntimeError: invalid multinomial distribution (encountering probability entry < 0)
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()
