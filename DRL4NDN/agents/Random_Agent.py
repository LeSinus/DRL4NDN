import random
import gc
import sys
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import os
from tqdm import tqdm
import argparse

# Add the parent directory to sys.path to allow importing from there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from helpers import train_agent, eval_test_agent
from gym_ndn.envs import NDNRouterEnv


def generate_color_palette():
    # Use tab20c which has 20 distinct colors and complement with tab20b
    tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    # Combine and select the first 30 unique colors
    all_colors = np.vstack((tab20c, tab20b))
    unique_colors = np.unique(all_colors, axis=0)[:30]
    return unique_colors


def plot_actions_over_time(timesteps_list, observations_list, actions_list, reward_func, state, scenario, num_faces,
                           seed,
                           draw_interval=30):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 18,
            }

    num_actions = max(actions_list) + 1  # compute the number of unique actions
    num_prefixes = int(max(observations_list) + 1)  # Compute the number of unique prefixes

    # Generate a list of colors based on the number of actions
    #cmap = plt.get_cmap('viridis')

    #colors = [cmap(i) for i in np.linspace(0, 1, num_actions)]

    # Generate a coherent set of 30 distinct colors
    colors = generate_color_palette()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Set y-tick labels for the observed prefixes
    y_tick_labels = [f'{i}' for i in range(num_prefixes)]
    ax.set_yticks(range(num_prefixes))
    ax.set_yticklabels(y_tick_labels)

    for action in set(actions_list):  # Iterating over each unique action
        # Extract only indices that match the draw_interval
        filtered_indices = [i for i in range(len(actions_list)) if actions_list[i] == action and i % draw_interval == 0]
        timesteps = [timesteps_list[i] for i in filtered_indices]
        observations = [observations_list[i] for i in filtered_indices]
        ax.scatter(timesteps, observations, color=colors[action % len(colors)], label=f'Face {action}', s=200)

    #ax.set_title('Actions Over Time')
    ax.set_xlabel('Timestep', fontdict=font)
    ax.set_ylabel('Requested packet prefixes', fontdict=font)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=15)

    legend = ax.legend(title="Faces", loc='upper left', bbox_to_anchor=(1, 1), fontsize='large',
                       title_fontsize='large')
    ###############################

    if num_faces == 30:
        plt.tight_layout(rect=[0, 0, 0.95, 1.3])  # Adjust the layout to make room for legend
    elif num_faces == 10:
        plt.tight_layout(rect=[0, 0, 0.95, 0.8])

    elif num_faces == 3:
        plt.tight_layout(rect=[0, 0, 0.95, 0.6])

    # Ensure the directory exists before saving
    save_dir = f'../logs/{scenario}/{reward_func}/{state}/RandomAgent/{num_faces}faces/{seed}seed/'
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(f'{save_dir}RandomAgent-{num_faces}-{seed}-Actions.pdf', format='pdf', dpi=1200, bbox_inches='tight',
                transparent=True)
    # plt.savefig(f'{save_dir}RandomAgent-{num_faces}-{seed}-Actions.svg', format='svg', dpi=1200, bbox_inches='tight',
    #             transparent=True)

    # plt.show()

    plt.close(fig)  # Close the plot to free up memory
    gc.collect()  # Collect garbage to free up memory


def RandomAgent(reward_func, state, scenario, face):
    # Logger configuration used both inside and outside NDNRouterEnv
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'../logs/{scenario}/{reward_func}/{state}/RandomAgent', exist_ok=True)
    file_name = f'../logs/{scenario}/{reward_func}/{state}/RandomAgent/RandomAgent{current_time}.log'
    # Configure a file handler for logging
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.INFO)

    # Create a logger instance
    logger = logging.getLogger('NDN_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    seeds = list(np.arange(0, 5))
    seeds = [int(x) for x in seeds]

    total_timesteps = 1000
    num_faces = face
    print(f"\nFaces: {num_faces}")

    # Iterate over different seeds
    total_rewards_per_seed = []
    for seed in tqdm(seeds):
        print(f"\nSeed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = gym.make('NDNRouter-v0', num_faces=num_faces, seed=seed, max_episode_length=1000, enable_logging=True,
                       logger=logger, state=state, reward_fct=reward_func, scenario=scenario)

        logger.info(f"\nEvaluating a Random Agent [Faces {num_faces} - Seed {seed}]:")

        total_rewards = 0
        timesteps_list = []
        actions_list = []
        observations_list = []
        obs, _ = env.reset()

        for current_step in range(total_timesteps):
            timesteps_list.append(current_step)
            observations_list.append(obs[0])
            action = env.action_space.sample()
            actions_list.append(action)
            obs, reward, done, _, _ = env.step(action)
            total_rewards += reward

        total_rewards_per_seed.append(total_rewards)
        print(f"Total reward of the Random Agent [Faces {num_faces} - Seed {seed}]: {total_rewards}")
        logger.info(f"Total reward of the Random Agent [Faces {num_faces} - Seed {seed}]: {total_rewards}")

        draw_interval = 1
        plot_actions_over_time(timesteps_list, observations_list, actions_list, reward_func, state, scenario, num_faces,
                               seed,
                               draw_interval)

        del env

    avg_reward = np.mean(total_rewards_per_seed)
    std_reward = np.std(total_rewards_per_seed)
    print(f"Average Reward [Faces {num_faces}]: {avg_reward} +/- {std_reward}")
    logger.info(f"\nAverage Reward [Faces {num_faces}]: {avg_reward} +/- {std_reward}")

    del logger
    gc.collect(0)


# @profile
def main():
    parser = argparse.ArgumentParser(description='Run BestAgent with different states and scenarios.')
    parser.add_argument('--reward', type=str, help='reward to use')
    parser.add_argument('--state', type=str, help='State to use')
    parser.add_argument('--scenario', type=str, help='Scenario to use')
    parser.add_argument('--face', type=int, help='face to use')

    args = parser.parse_args()
    RandomAgent(args.reward, args.state, args.scenario, args.face)


if __name__ == "__main__":
    main()
