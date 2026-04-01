import sys
import os
import random
import torch
import gc
import gymnasium as gym
import numpy as np
import logging
from datetime import datetime
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import TRPO, QRDQN
import os
import argparse
# from tqdm import tqdm

# Add the parent directory to sys.path to allow importing from there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from helpers import train_agent, eval_test_agent
from gym_ndn.envs import NDNRouterEnv

seeds = list(np.arange(0, 5))
# seeds = list(np.setdiff1d(np.arange(0, 101), [79]))
seeds = [int(x) for x in seeds]

nb_faces_params = {3: (30000, 1000), 10: (100000, 5000), 30: (1000000, 10000)}

# nb_faces_params = {3: (10000, 1000), 10: (50000, 5000), 30: (100000, 10000)}
# nb_faces_params = {3: (30000, 1000), 10: (100000, 5000), 30: (1000000, 10000)}

# nb_faces_params = {3: (100000, 1000), 10: (500000, 5000), 30: (2000000, 20000)}
# nb_faces_params = {3: (10000, 1000), 10: (50000, 5000), 30: (100000, 10000)}
# nb_faces_params = {10: (50000, 5000), 30: (100000, 10000)}
# nb_faces_params = {3: (50000, 10000)}

device = "cpu"


def agents(reward, state, scenario, agent, face):
    default_agent_params = {
        PPO: {},
        A2C: {},
        DQN: {'learning_starts': 1000, 'gamma': 0.96, 'buffer_size': 10000,
              'target_update_interval': 50, 'batch_size': 64},
        TRPO: {},
        QRDQN: {'learning_starts': 1000, 'gamma': 0.96, 'buffer_size': 10000,
                'target_update_interval': 50, 'batch_size': 64},
    }
    # Logger configuration used both inside and outside NDNRouterEnv
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'../logs/{scenario}/{reward}/{state}/DRL Agents/{agent.__name__}', exist_ok=True)
    file_name = f'../logs/{scenario}/{reward}/{state}/DRL Agents/{agent.__name__}/{agent.__name__}_{current_time}.log'
    # Configure a file handler for logging
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)

    # Create a logger instance
    logger = logging.getLogger('NDN_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    num_faces = face

    print(f"\nFaces: {num_faces}")

    # Unpack the parameters based on the number of faces
    training_budget, training_eval_freq = nb_faces_params[num_faces]
    total_rewards_per_seed = []
    # Iterate over different seeds
    # for seed in tqdm(seeds):
    for seed in seeds:
        print(f"\nSeed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"\nTraining and testing {agent.__name__} [Faces {num_faces} - Seed {seed}] ...")
        log_dir = f'../logs/{scenario}/{reward}/{state}/DRL Agents/{agent.__name__}/{num_faces}faces/{seed}seed'
        os.makedirs(log_dir, exist_ok=True)

        train_env = gym.make('NDNRouter-v0', num_faces=num_faces, seed=seed, max_episode_length=1000,
                             enable_logging=False, state=state, reward_fct=reward, scenario=scenario)

        eval_env = gym.make('NDNRouter-v0', num_faces=num_faces, seed=seed, max_episode_length=1000,
                            enable_logging=True,
                            logger=logger, state=state, reward_fct=reward, scenario=scenario)

        agent_params = default_agent_params[agent].copy()

        # Train and test the agent
        best_model = train_agent(agent, train_env, device, agent_params, num_faces, seed, logger, log_dir,
                                 total_timesteps=training_budget, eval_freq=training_eval_freq,
                                 n_eval_episodes=10)

        total_rewards = eval_test_agent(agent, eval_env, best_model, 10, logger, 1000,
                                        num_faces, seed, log_dir)

        total_rewards_per_seed.append(total_rewards)
        print(f"\n{agent.__name__} [Faces {num_faces} - Seed {seed}] training and testing completed.")

        del train_env, eval_env, agent_params
        gc.collect()

    avg_reward = np.mean(total_rewards_per_seed)
    std_reward = np.std(total_rewards_per_seed)
    print(f"Average Reward [Faces {num_faces}]: {avg_reward} +/- {std_reward}")
    logger.info(f"\nAverage Reward [Faces {num_faces}]: {avg_reward} +/- {std_reward}")
    del logger


def main(reward, state, scenario, agent, face):
    agents(reward, state, scenario, agent, face)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run different agents on various scenarios.')
    parser.add_argument('--reward', type=str, help='reward of the simulation')
    parser.add_argument('--state', type=str, help='State of the simulation')
    parser.add_argument('--scenario', type=str, help='Scenario to simulate')
    parser.add_argument('--agent', type=str, help='Agent to use (PPO, A2C, DQN, TRPO, QRDQN)')
    parser.add_argument('--face', type=int, help='face to use')

    args = parser.parse_args()

    agent_classes = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN, 'TRPO': TRPO, 'QRDQN': QRDQN}
    agent_class = agent_classes.get(args.agent)

    if agent_class:
        main(args.reward, args.state, args.scenario, agent_class, args.face)
    else:
        print(f"Agent {args.agent} is not recognized.")

    # reward = "simple"
    # state = "state_two"
    # scenario = "Dynamic_RTT"
    # agent_classes = {'PPO': PPO, 'A2C': A2C, 'DQN': DQN, 'TRPO': TRPO, 'QRDQN': QRDQN}
    #
    # agent_class = agent_classes.get('DQN')
    #
    # if agent_class:
    #     main(reward, state, scenario, agent_class)
    # else:
    #     print(f"Agent is not recognized.")
