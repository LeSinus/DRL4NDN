import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.monitor import Monitor
import pandas as pd

"""
# helper function that evaluate an agent and shows its policy,
# i.e., selected faces as a function of the interest prefix
def test_agent(agent, environment, total_timesteps=1000, draw_interval=10):
    timesteps_list = []
    actions_list = []
    observations_list = []
    total_rewards = 0

    obs, _ = environment.reset()

    for current_step in range(total_timesteps):
        timesteps_list.append(current_step)
        observations_list.append(obs[0])
        action, _ = agent.predict(obs, deterministic=True)
        actions_list.append(action)
        obs, reward, done, _, info = environment.step(action)
        total_rewards += reward

    agent_name = type(agent).__name__  # Extract the agent's name
    print(f"Total reward of {agent_name}: {total_rewards}")

    # Plotting
    plt.scatter(timesteps_list[::draw_interval], observations_list[::draw_interval],
                c=actions_list[::draw_interval], cmap='viridis', marker='o')
    plt.colorbar()
    plt.title(f'{agent_name} Actions Over Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Observations')
    plt.show()

    environment.close()


# An enhanced version of the previous function
def test_agent2(agent_model, environment, total_timesteps=1000, draw_interval=10, log_dir="./logs/"):
    timesteps_list = []
    actions_list = []
    observations_list = []
    total_rewards = 0

    obs, _ = environment.reset()

    for current_step in range(total_timesteps):
        timesteps_list.append(current_step)
        observations_list.append(obs[0])
        # As some policies are stochastic by default (e.g. A2C or PPO), you
        # should also try to set deterministic=True when calling the .predict()
        # method, this frequently leads to better performance.
        action, _ = agent_model.predict(obs, deterministic=True)
        actions_list.append(action)
        obs, reward, done, _, info = environment.step(action)
        total_rewards += reward

    agent_name = type(agent_model).__name__  # Extract the agent's name
    print(f"Total reward of {agent_name}: {total_rewards}")

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(timesteps_list[::draw_interval], observations_list[::draw_interval],
                          c=actions_list[::draw_interval], cmap='viridis', marker='o')

    # Add custom legend for faces
    handles, labels = scatter.legend_elements(prop='colors', alpha=0.6)
    legend_labels = [f'Face {i}' for i in range(environment.unwrapped.num_faces)]
    cbar = plt.colorbar(scatter, ticks=range(environment.unwrapped.num_faces))
    cbar.set_label('Faces')
    cbar.set_ticklabels(legend_labels)

    # Enhance y-axis labels to show prefixes
    prefix_labels = [f'Prefix{i}' for i in environment.unwrapped.interests]
    plt.yticks(environment.unwrapped.interests, prefix_labels)

    plt.title(f'{agent_name} Actions Over Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Observations')
    # plt.savefig(f'{log_dir}/{agent_name}-policy.eps', format="eps", dpi=1200, bbox_inches="tight", transparent=True)
    plt.savefig(f'{log_dir}/{agent_name}-policy.pdf', format="pdf", dpi=1200, bbox_inches="tight", transparent=True)
    plt.close('all')
    # plt.show()
    del timesteps_list, actions_list, observations_list, scatter
    gc.collect(2)
    environment.close()


def test_agent_fast(agent_model, environment, total_timesteps=1000, draw_interval=10):
    timesteps_list = []
    actions_list = []
    observations_list = []
    total_rewards = 0

    obs, _ = environment.reset()

    for current_step in range(total_timesteps):
        timesteps_list.append(current_step)
        observations_list.append(obs[0])
        action, _ = agent_model.predict(obs, deterministic=True)
        actions_list.append(action)
        obs, reward, done, _, info = environment.step(action)
        total_rewards += reward

    agent_name = type(agent_model).__name__  # Extract the agent's name
    print(f"Total reward of {agent_name}: {total_rewards}")

    del timesteps_list, actions_list, observations_list
    gc.collect(2)
    environment.close()


def train_and_test_agent(agent_class, train_env, eval_env, device, agent_params, logger, num_faces, seed,
                         total_timesteps, eval_freq, n_eval_episodes):
    # Extract the base model name from the agent class
    base_model_name = agent_class.__name__

    # Create log dir
    log_dir = f'./logs/{base_model_name}/{num_faces}faces/{seed}seed'
    os.makedirs(log_dir, exist_ok=True)

    # Wrap the environment
    # Logs will be saved in log_dir/monitor.csv
    train_env_m = Monitor(train_env, log_dir)

    # Initialize the agent with custom parameters
    model = agent_class("MlpPolicy", train_env_m, verbose=0, device=device, **agent_params)

    # Configure the callback for evaluation with logging
    eval_callback = EvalCallback(train_env_m,
                                 best_model_save_path=log_dir,
                                 log_path=log_dir,
                                 eval_freq=eval_freq,
                                 n_eval_episodes=5,
                                 deterministic=True,
                                 render=False)

    # Train the agent with the callback
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Load the best saved model
    best_model = agent_class.load(f'{log_dir}/best_model')

    # Evaluate the trained agent
    # Compute and log mean_reward, std_reward, and the packet loss ratio
    logger.info(f"Evaluating the {base_model_name} Agent [{num_faces}-{seed}]: {n_eval_episodes}")
    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes, warn=False)
    logger.info(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # display the policy
    test_agent2(best_model, train_env, 1000, 10, log_dir)
    del model, best_model, train_env_m
    gc.collect(2)


def train_and_test_agent_fast(agent_class, train_env, eval_env, device, agent_params, logger, num_faces, seed,
                              total_timesteps, n_eval_episodes):
    # Extract the base model name from the agent class
    base_model_name = agent_class.__name__

    # Initialize the agent with custom parameters
    model = agent_class("MlpPolicy", train_env, verbose=0, device=device, **agent_params)

    # Train the agent with the callback
    model.learn(total_timesteps=total_timesteps)

    # Evaluate the trained agent
    # Compute and log mean_reward, std_reward, and the packet loss ratio
    logger.info(f"Evaluating the {base_model_name} Agent [{num_faces}-{seed}]: {n_eval_episodes}")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes, warn=False)
    logger.info(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # display the policy
    # test_agent_fast(model, train_env, 1000, 10)
    del model
    gc.collect(2)

"""


def plot_evaluations_data(log_dir):
    # Path to the evaluations.npz file
    results_dir = log_dir + "/results"
    file_path = f"{results_dir}/evaluations.npz"
    save_dir = log_dir + "/plots"
    os.makedirs(save_dir, exist_ok=True)
    # Load the data
    data = np.load(file_path)

    # The file contains multiple arrays:
    # - 'timesteps': the number of timesteps at which each evaluation was performed
    # - 'results': the rewards obtained in each evaluation
    # - 'ep_lengths': the lengths of episodes in each evaluation
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']

    # Calculate average rewards and episode lengths
    avg_rewards = np.mean(results, axis=1)  # Average over all episodes in each evaluation
    avg_ep_lengths = np.mean(ep_lengths, axis=1)

    # Plotting average rewards over timesteps
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, avg_rewards, marker='o', color='b', label='Average Reward')
    plt.title('Average Reward over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()
    # Save the figure as a PDF in the log directory
    plt.savefig(f"{save_dir}/evaluation_rewards.pdf", format='pdf', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()  # Close the plot to free up memory
    # plt.show()

    # Plotting average episode lengths over timesteps
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, avg_ep_lengths, marker='o', color='r', label='Average Episode Length')
    plt.title('Average Episode Length over Time')
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Length')
    plt.grid(True)
    plt.legend()
    # Save the figure as a PDF in the log directory
    plt.savefig(f"{save_dir}/evaluation_Episode_Length.pdf", format='pdf', dpi=300, bbox_inches='tight',
                transparent=True)
    plt.close()  # Close the plot to free up memory
    # plt.show()


def plot_monitor_data(log_dir):
    results_dir = log_dir + "/results"
    file_path = f"{results_dir}/monitor.csv"
    save_dir = log_dir + "/plots"
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(file_path, skiprows=1)  # Skipping the first row which is often headers

    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(df['r'], label='Rewards', color='blue')
    ax1.set_title('Rewards over Time')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward')
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/monitor_rewards_over_time.pdf", format='pdf', dpi=300)
    plt.close(fig1)  # Close the plot to free up memory

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(df['l'], label='Episode Lengths', color='red')
    ax2.set_title('Episode Lengths over Time')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Length')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/monitor_episode_lengths.pdf", format='pdf', dpi=300)
    plt.close(fig2)  # Close the plot to free up memory

    del df
    # plt.show()


def generate_color_palette():
    # Use tab20c which has 20 distinct colors and complement with tab20b
    tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    # Combine and select the first 30 unique colors
    all_colors = np.vstack((tab20c, tab20b))
    unique_colors = np.unique(all_colors, axis=0)[:30]
    return unique_colors


def plot_actions_over_time(agent_class, timesteps_list, observations_list, actions_list, num_faces,
                           seed, log_dir,
                           draw_interval=5):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 18,
            }

    save_dir = log_dir + "/plots"
    os.makedirs(save_dir, exist_ok=True)

    base_model_name = agent_class.__name__
    num_actions = max(actions_list) + 1  # compute the number of unique actions
    num_prefixes = int(max(observations_list) + 1)  # Compute the number of unique prefixes

    # Generate a list of colors based on the number of actions
    # cmap = plt.get_cmap('viridis')
    # colors = [cmap(i) for i in np.linspace(0, 1, num_actions)]

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

    ax.set_xlabel('Timestep', fontdict=font)
    ax.set_ylabel('Requested packet prefixes', fontdict=font)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # Position the legend outside of the figure on the right, and adjust the size and placement precisely
    legend = ax.legend(title="Faces", loc='upper left', bbox_to_anchor=(1, 1), fontsize='large',
                       title_fontsize='large')
    if num_faces == 30:
        plt.tight_layout(rect=[0, 0, 0.95, 1.3])  # Adjust the layout to make room for legend
    elif num_faces == 10:
        plt.tight_layout(rect=[0, 0, 0.95, 0.8])

    elif num_faces == 3:
        plt.tight_layout(rect=[0, 0, 0.95, 0.6])

    # Save the plot
    plt.savefig(f'{save_dir}/{base_model_name}-{num_faces}-{seed}-Actions.pdf', format='pdf', dpi=1200,
                bbox_inches='tight',
                transparent=True)

    # plt.savefig(f'{save_dir}/{base_model_name}-{num_faces}-{seed}-Actions.svg', format='svg', dpi=1200,
    #             bbox_inches='tight',
    #             transparent=True)
    # plt.show()

    plt.close(fig)  # Close the plot to free up memory
    gc.collect()  # Collect garbage to free up memory


def train_agent(agent_class, train_env, device, agent_params, num_faces, seed, logger, log_dir, total_timesteps,
                eval_freq,
                n_eval_episodes):
    save_dir = log_dir + "/results"
    os.makedirs(save_dir, exist_ok=True)
    # Wrap the environment with a Monitor for logging
    train_env_m = Monitor(train_env, save_dir)

    # Initialize the agent with custom parameters
    model = agent_class("MlpPolicy", train_env_m, verbose=0, device=device, **agent_params)

    # Configure the callback for evaluation with logging
    eval_callback = EvalCallback(train_env_m,
                                 best_model_save_path=save_dir,
                                 log_path=save_dir,
                                 eval_freq=eval_freq,
                                 n_eval_episodes=n_eval_episodes,
                                 deterministic=True,
                                 render=False)

    # Train the agent with the callback
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save the trained model
    model.save(f'{save_dir}/final_model')
    # Load the best saved model
    best_model = agent_class.load(f'{save_dir}/best_model')

    logger.info(f"\nTraining completed for {agent_class.__name__} Agent [{num_faces}-{seed}]")

    return best_model


def eval_test_agent(agent_class, eval_env, best_model, n_eval_episodes, logger, total_timesteps,
                    num_faces, seed, log_dir):
    agent_name = agent_class.__name__
    print(f" Evaluate the {agent_name} agent")

    logger.info(f"\nEvaluating the {agent_name} Agent [{num_faces}-{seed}]:"
                f" with evaluate_policy in {n_eval_episodes} episodes")
    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes, warn=False)
    logger.info(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # logger.info(f"\nEvaluating the {agent_name} Agent [{num_faces}-{seed}] Man")
    timesteps_list = []
    actions_list = []
    observations_list = []
    total_rewards = 0

    obs, _ = eval_env.reset()

    for current_step in range(total_timesteps):
        timesteps_list.append(current_step)
        observations_list.append(obs[0])
        action, _ = best_model.predict(obs, deterministic=True)
        actions_list.append(int(action))
        obs, reward, done, _, info = eval_env.step(action)
        total_rewards += reward

    # print(f"Total reward of the {agent_name} agent [Faces {num_faces} - Seed {seed}]: {total_rewards}")
    # logger.info(f"Total reward of the {agent_name} agent [Faces {num_faces} - Seed {seed}]: {total_rewards}")

    draw_interval = 1

    plot_actions_over_time(agent_class, timesteps_list, observations_list, actions_list, num_faces,
                           seed, log_dir,
                           draw_interval)

    plot_monitor_data(log_dir)
    plot_evaluations_data(log_dir)

    del timesteps_list, actions_list, observations_list
    gc.collect(2)  # Collect garbage to free up resources
    return total_rewards
