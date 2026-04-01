import sys
import os
import gc
import gymnasium as gym
import numpy as np
import logging
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import random
import torch

# Add the parent directory to sys.path to allow importing from there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from helpers import train_agent, eval_test_agent, plot_actions_over_time
from gym_ndn.envs import NDNRouterEnv
from MultiArmedBandit import LinUCB

seeds = list(np.arange(0, 5))
seeds = [int(x) for x in seeds]
# nb_faces = [3, 10, 30]
# nb_faces = [3]
total_timesteps = 1000

cmab_agents = [LinUCB]

device = "cpu"


def agents(reward_func, state, scenario, agent, face):
    # Logger configuration used both inside and outside NDNRouterEnv
    global context_dim
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'../logs/{scenario}/{reward_func}/{state}/CMAB Agents/{agent.__name__}/', exist_ok=True)
    file_name = f'../logs/{scenario}/{reward_func}/{state}/CMAB Agents/{agent.__name__}/{agent.__name__}_{current_time}.log'
    # Configure a file handler for logging
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)

    # Create a logger instance
    logger = logging.getLogger('NDN_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    num_faces = face
    print(f"\nFaces: {num_faces}")
    # Initialize CMulti-Armed Bandit
    if state == "state_one":
        context_dim = num_faces * 6 + 1
    elif state == "state_two":
        context_dim = num_faces * 2 + 1
    elif state == "state_three":
        context_dim = num_faces * 2 + 1
    elif state == "state_four":
        context_dim = num_faces * 4 + 1

    print(f"context_dim:{context_dim}")
    cmab = agent(num_faces, context_dim)

    print(f"Evaluating the {cmab.__class__.__name__} Agent [Faces {num_faces}]")
    total_rewards_per_seed = []
    # Iterate over different seeds
    for seed in tqdm(seeds):
        print(f"\nSeed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        log_dir = f'../logs/{scenario}/{reward_func}/{state}/CMAB Agents/{agent.__name__}/{num_faces}faces/{seed}seed'
        os.makedirs(log_dir, exist_ok=True)

        logger.info(f"\nEvaluating the {cmab.__class__.__name__} Agent [{num_faces}-{seed}]:")

        train_env = gym.make('NDNRouter-v0', num_faces=num_faces, seed=seed, state=state, reward_fct=reward_func,
                             scenario=scenario)

        eval_env = gym.make('NDNRouter-v0', num_faces=num_faces, seed=seed, enable_logging=True, logger=logger,
                            state=state, reward_fct=reward_func, scenario=scenario)

        total_rewards = 0
        timesteps_list = []
        actions_list = []
        observations_list = []
        obs, _ = train_env.reset()

        # training
        for current_step in range(100):
            context = obs
            action_cmab = cmab.select_arm(context)
            obs, reward_cmab, _, _, _ = train_env.step(action_cmab)
            cmab.update_arm(action_cmab, context, reward_cmab)

        # evaluation
        obs, _ = eval_env.reset()
        for current_step in range(total_timesteps):
            timesteps_list.append(current_step)
            observations_list.append(obs[0])
            context = obs  # the context is the observation
            action_cmab = cmab.select_arm(context)
            actions_list.append(action_cmab)
            obs, reward_cmab, _, _, _ = eval_env.step(action_cmab)
            cmab.update_arm(action_cmab, context, reward_cmab)
            total_rewards += reward_cmab

        total_rewards_per_seed.append(total_rewards)
        print(
            f"Total reward of the {cmab.__class__.__name__} Agent [Faces {num_faces} - Seed {seed}]: {total_rewards}")
        logger.info(
            f"Total reward of the {cmab.__class__.__name__} Agent [{num_faces}-{seed}]: {total_rewards}")

        draw_interval = 1

        plot_actions_over_time(agent_class, timesteps_list, observations_list, actions_list, num_faces,
                               seed, log_dir, draw_interval)

        del train_env, eval_env
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
    parser.add_argument('--agent', type=str, help='Agent to use (LinUCB)')
    parser.add_argument('--face', type=int, help='face to use')

    args = parser.parse_args()

    agent_classes = {'LinUCB': LinUCB}

    agent_class = agent_classes.get(args.agent)

    if agent_class:
        main(args.reward, args.state, args.scenario, agent_class, args.face)
    else:
        print(f"Agent {args.agent} is not recognized.")

    # reward = "simple"
    # state = "state_three"
    # scenario = "static_with_one_optimal_face"
    # agent_classes = {'LinUCB': LinUCB}
    #
    # agent_class = agent_classes.get('LinUCB')
    #
    # if agent_class:
    #     main(reward, state, scenario, agent_class)
    # else:
    #     print(f"Agent is not recognized.")
