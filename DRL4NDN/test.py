import numpy as np
import gymnasium as gym
from gym_ndn.envs import NDNRouterEnv


# Initialize the custom NDNRouterEnv
state = 'state_one'  # Or whichever state you want to test with
reward_fct = 'simple'  # Or 'streak', based on your requirement
scenario = 'static_with_one_optimal_face'  # Choose any scenario you want to test
seed = 79
num_faces = 3
max_episode_length = 1000
render_mode = None
enable_logging = False
logger = None  # Pass a logger object if you want to enable logging

env = NDNRouterEnv(state, reward_fct, scenario, seed, num_faces, max_episode_length, render_mode, enable_logging,
                   logger)

# Reset the environment to start
observation, info = env.reset()

# Print initial observation and info
print(f"Initial observation: {observation}")
# print(f"Initial info: {info}")

# Run a simple loop to take random actions and print the results
num_steps = 1000  # Number of steps to simulate
total_reward = 0

for step in range(num_steps):
    # Update the scenario dynamics first
    action = env.unwrapped.get_best_face()
    np.random.rand()
    # Take the action in the environment
    observation, reward, done, _, _ = env.step(action)

    # Print the results of the action
    print(f"Step {step + 1} | Action: {action} | Reward: {reward}")
    # print(f"  Observation: {observation}")
    # print(f"  Done: {done}")
    # print(f"  Info: {info}")
    total_reward += reward

    # Check if the episode is done
    if done:
        print("Episode finished.")
        break


print(f"Total reward after {step + 1} steps: {total_reward}")