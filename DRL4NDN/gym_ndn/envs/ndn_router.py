import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import sys
from collections import deque
import copy

import math

# Constants
MEAN_RTT_FACES = 100
STANDARD_DEVIATION_MEAN_RTT_FACES = 30
RTT_VARIABILITY_FACES = 5
NACK_NEGATIVE_REWARD = TIMEOUT_NEGATIVE_REWARD = -1.0
DEFAULT_TIMEOUT_PROBABILITY = 0.0
DEFAULT_NACK_PROBABILITY = 0.0
MOVING_WINDOW_SIZE = 50


class NDNRouterEnv(gym.Env):
    """
    Custom Gym environment representing a Named Data Networking (NDN) router.

    This environment simulates a router with multiple faces (interfaces) that forward interests (packets)
    and receive responses.

    Attributes:
        metadata (dict): Metadata for the environment.
        local_random_np: a local numpy random state.
        local_random: a local random state.
        seed (int): Seed for random number generation.
        num_faces (int): Number of faces (interfaces) in the router.
        face_prefix_sensitivity (bool): Flag indicating whether faces exhibit sensitivity to specific prefixes
        enable_logging (bool): Flag indicating whether logging is enabled.
        action_space (gym.Space): Action space definition for the environment.
        max_episode_length (int): Maximum length of an episode.
        observation_space (gym.Space): Observation space definition for the environment.
        interests (np.ndarray): Array representing interests handled by the router.
        current_step (int): Current step count in the environment.
        means (np.ndarray): Array representing RTT means for each face.
        stDev (float): Standard deviation for RTT distribution.
        nack_probability (dict): Dictionary representing Nack probability for each face.
        timeout_probability (dict): Dictionary representing Timeout probability for each face.
        router_state (dict): Dictionary representing state information for each face.
        current_prefix (int): Current prefix being processed by the router.
        selected_face (str): Selected face for the current action.
        render_mode (str): Rendering mode for visualization ('human' or 'rgb_array').
        logger (Logger or None): Logger object for logging environment events.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    __slots__ = ['local_random_np', 'local_random', 'seed', 'num_faces', 'face_prefix_sensitivity', 'enable_logging',
                 'action_space', 'max_episode_length', 'observation_space', 'interests', 'current_step', 'means',
                 'max_RTTs', 'stDev', 'nack_probability', 'timeout_probability', 'router_state', 'current_prefix',
                 'selected_face', 'render_mode', 'logger', 'state', 'scenario', 'reward_fct',
                 'success_streak', 'failure_streak']

    def __init__(self, state, reward_fct,
                 scenario, seed=0, num_faces=3, max_episode_length=1000, render_mode=None,
                 enable_logging=False, logger=None):
        super(NDNRouterEnv, self).__init__()

        self.available_scenarios = [
            "static_with_one_optimal_face",
            "dynamic_RTT",
            "permanent_face_specific",
            "transient_face_specific",
            "permanent_faulty_face",
            "transient_faulty_face",
            "streak_disruption",
        ]
        # Number of faces: [N.B. A virtual face could be added to account for interest suppression]
        self.num_faces = num_faces

        # prefixes considered: [0, 1, 2, 3, 4, ... num_faces-1]
        self.interests = np.arange(0, self.num_faces, 1)

        self.scenario = scenario
        self.state = state
        self.reward_fct = reward_fct
        self.success_streak = 0  # Initialize a counter for successful consecutive responses
        self.failure_streak = 0  # Initialize a counter for unsuccessful consecutive responses

        self.faces_data = self._initialize_faces_data()

        # set the seeds for reproducibility
        self.seed = seed
        self.local_random_np = np.random.RandomState(self.seed)
        self.local_random = random.Random(self.seed)


        # Initialize prefix sensitivity: map each face to be able to handle all prefixes
        self.face_prefix_sensitivity = {f'face{i}': list(self.interests) for i in range(self.num_faces)}
        self.enable_logging = enable_logging

        # Define action space: [what the agent can do {choose face1 or face2 or face3 ...}]
        self.action_space = spaces.Discrete(self.num_faces)

        # Maximum episode length
        self.max_episode_length = max_episode_length

        # Define observation spaces
        # Several metrics per face + prefix
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                            shape=(self.num_faces * len(self._init_face_state()) + 1,),
                                            dtype=np.float32)

        # Current step count
        self.current_step = 0

        # Initialize the first received interest prefix
        self.current_prefix = 0
        self.selected_face = 0

        # Our statistical back-end based on normal distributions
        # RTT distributions (use abs to enforce positive values)
        self.means = abs(self.local_random_np.normal(MEAN_RTT_FACES, STANDARD_DEVIATION_MEAN_RTT_FACES, self.num_faces))
        self.stDev = RTT_VARIABILITY_FACES
        self.max_RTTs = np.ceil(self.means + 3 * self.stDev)

        # Probability of generating Nack or Timeout for each face
        self.nack_probability = {f'face{i}': DEFAULT_NACK_PROBABILITY for i in range(0, self.num_faces)}
        self.timeout_probability = {f'face{i}': DEFAULT_TIMEOUT_PROBABILITY for i in range(0, self.num_faces)}

        # Initialize router state for each face
        self.router_state = {f'face{i}': self._init_face_state() for i in range(0, self.num_faces)}

        self.scenario_dynamic()

        self._best_face_index = self.get_best_face()  # Store the index of the best face

        # logging
        if logger is not None:
            self.logger = logger

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _initialize_faces_data(self):
        # Initialize the faces_data dictionary with default data for each face
        initial_face_data = {
            'sentInterests': 0,
            'receivedData': 0,
            'MovingSentInterests': 0,
            'MovingReceivedData': 0,
            'receivedNack': 0,
            'nbTimeout': 0,
            'avgRTT': 0.0,
            'lastRTT': 0.0,
            'prefixSuccesses': {prefix: 0 for prefix in self.interests},
            'prefixAttempts': {prefix: 0 for prefix in self.interests},
            'movingAvgRTTPrefix': {prefix: deque(maxlen=MOVING_WINDOW_SIZE) for prefix in self.interests}  # A queue for each prefix
        }
        faces_data = {f'face{i}': copy.deepcopy(initial_face_data) for i in range(self.num_faces)}
        return faces_data

    def _get_action_space(self):
        return self.action_space

    def _get_observation_space(self):
        return self.observation_space

    def show_router_settings(self):
        """
        Displays the configuration of the router.

        This method prints various configuration parameters of the router, such as the number of faces,
        Nack probabilities, Timeout probabilities, RTT means, and prefix sensitivity settings.
        """
        print(f"Number of faces: {self.num_faces}")
        print(f"Nack Probabilities: {self.nack_probability}")
        print(f"Timeout Probabilities: {self.timeout_probability}")
        print(f"Faces RTT Means: {self.means}")

        # Check if face_prefix_sensitivity is a dictionary and print accordingly
        if isinstance(self.face_prefix_sensitivity, dict):
            print("Face Prefix Sensitivity Settings:")
            for face, prefixes in self.face_prefix_sensitivity.items():
                print(f"  {face}: {prefixes}")
        else:
            print(f"Faces are sensitive to specific prefixes: {self.face_prefix_sensitivity}")

        # Optionally, add additional configuration details
        print(f"Render Mode: {self.render_mode}")
        print(f"Current Simulation Step: {self.current_step}")

    def get_best_face(self):
        """
        Determines the index of the face with the optimal RTT mean, considering the current scenario.
        """

        if self.scenario in ["static_with_one_optimal_face", "dynamic_RTT"]:
            return np.argmin(self.means)

        elif self.scenario in ["permanent_faulty_face", "transient_faulty_face"]:
            valid_faces = [
                i for i in range(self.num_faces)
                if (self.nack_probability[f'face{i}'] + self.timeout_probability[f'face{i}']) < 1
            ]
            # Find the face with the minimum RTT mean among valid faces
            return min(valid_faces, key=lambda i: self.means[i]) if valid_faces else None

        elif self.scenario in ["permanent_face_specific", "transient_face_specific"]:
            valid_faces = [
                i for i in range(self.num_faces)
                if self.current_prefix in self.face_prefix_sensitivity[f'face{i}']
                   and (self.nack_probability[f'face{i}'] + self.timeout_probability[f'face{i}']) < 1
            ]
            return min(valid_faces, key=lambda i: self.means[i]) if valid_faces else None

        elif self.scenario in ["streak_disruption"]:
            valid_faces = [
                i for i in range(self.num_faces)
                if self.current_prefix in self.face_prefix_sensitivity[f'face{i}']
                   and (self.nack_probability[f'face{i}']) < 1.0
            ]
            # print("valid_faces", valid_faces)
            # print("best face", min(valid_faces, key=lambda i: self.means[i]) if valid_faces else None)
            return min(valid_faces, key=lambda i: self.means[i]) if valid_faces else None

        else:
            print(f"Unknown scenario: {self.scenario}")
            return None

    # TODO second best face scenario-dependent
    def get_second_best_face(self):
        """
        Determines the index of the face with the second minimum RTT mean.

        Returns:
            int: Index of the face with the second minimum RTT mean.
        """
        index_of_min = np.argmin(self.means)
        index_of_second_min = np.argmin(np.where(np.arange(self.num_faces) != index_of_min, self.means, np.inf))
        return index_of_second_min

    def _init_face_state(self):
        """
        Initializes the state for each face.

        Returns:
            dict: Dictionary representing the initial state for a face.
        """
        if self.state == 'state_one':
            return {
                'sentInterests': 0,
                'receivedData': 0,
                'receivedNack': 0,
                'nbTimeout': 0,
                'avgRTT': 0.0,
                'lastRTT': 0.0
            }
        elif self.state == 'state_two':
            return {
                'successRate': 0.0,  # Success rate for the face
                'movingAvgRTT': 0.0  # Overall moving average across all prefixes
            }
        elif self.state == 'state_three':

            return {
                'movingSuccessRate': 0.0,  # Success rate for the face
                'movingAvgRTT': 0.0  # Overall moving average across all prefixes
            }

        elif self.state == 'state_four':
            return {
                'movingSuccessRate': 0.0,  # Success rate for the face
                'movingAvgRTT': 0.0,  # Overall moving average across all prefixes
                'movingSuccessRatePerPrefix': 0.0,  # Success rate for the face per current prefix
                'movingAvgRTTPerPrefix': 0.0,  # moving average on current prefix
            }

    def _get_obs(self):
        """
        Retrieves the observation representing the current router state.

        Returns:
            np.ndarray: Observation array representing the current router state for all faces and the current prefix.

        """
        # Return the current router state as the observation for all faces + prefix
        obs = [float(self.current_prefix)]
        for face in self.router_state:
            obs.extend([float(value) for value in self.router_state[face].values()])
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):

        # Report total_data_received from the previous episode using the logger
        if self.enable_logging:
            total_interest_sent = sum(self.faces_data[face]['sentInterests'] for face in self.router_state)
            if total_interest_sent > 0:
                total_data_received = sum(self.faces_data[face]['receivedData'] for face in self.router_state)
                self.logger.info(
                    f"total_interest_sent: {total_interest_sent} | total_data_received: {total_data_received}")

                Average_RTT = sum(self.faces_data[face]['avgRTT'] for face in self.router_state) / self.num_faces

                self.logger.info(
                    f"Average RTT: {Average_RTT}")

        # Reset counters
        self.current_step = 0
        self.current_prefix = 0
        self.success_streak = 0
        self.failure_streak = 0

        # Reset faces data
        self.faces_data = self._initialize_faces_data()

        # Reset router state
        for face in self.router_state:
            self.router_state[face] = self._init_face_state()

        # Reset means
        self.local_random = random.Random(self.seed)
        self.local_random_np = np.random.RandomState(self.seed)
        self.means = abs(self.local_random_np.normal(MEAN_RTT_FACES, STANDARD_DEVIATION_MEAN_RTT_FACES, self.num_faces))

        # Reset max_RTTs
        self.max_RTTs = np.ceil(self.means + 3 * self.stDev)

        # Reset Nack Proba
        self.nack_probability = {f'face{i}': DEFAULT_NACK_PROBABILITY for i in range(0, self.num_faces)}

        # Reset Timeout Proba
        self.timeout_probability = {f'face{i}': DEFAULT_TIMEOUT_PROBABILITY for i in range(0, self.num_faces)}

        self.scenario_dynamic()

        # Reset Timeout proba

        if self.render_mode == "human":
            self._render_human()

        return self._get_obs(), self._get_info()

    def _determine_response_type(self, selected_face):
        # Calculate the probabilities of success, Nack, and Timeout based on the selected face
        pnack = self.nack_probability[selected_face]
        ptimeout = self.timeout_probability[selected_face]
        success = 1 - (pnack + ptimeout)

        # Choose the response type based on the calculated probabilities
        return self.local_random_np.choice(['RTT', 'Nack', 'Timeout'], p=[success, pnack, ptimeout])

    def _process_response(self, action, response_type):
        # Simulate the response based on the response type
        if response_type == 'RTT':
            self.success_streak += 1
            self.failure_streak = 0
            rtt = float(abs(self.local_random_np.normal(self.means[action], self.stDev)))
            normalized_rtt = rtt / max(self.max_RTTs)
            self.faces_data[f'face{action}']['prefixSuccesses'][self.current_prefix] += 1
            self.faces_data[f'face{action}']['lastRTT'] = normalized_rtt

            self.faces_data[f'face{action}']['movingAvgRTTPrefix'][self.current_prefix].append(normalized_rtt)
            self.faces_data[f'face{action}']['avgRTT'] = (
                                                                 self.faces_data[f'face{action}'][
                                                                     'avgRTT'] *
                                                                 self.faces_data[f'face{action}'][
                                                                     'receivedData'] + normalized_rtt) / (
                                                                 self.faces_data[f'face{action}'][
                                                                     'receivedData'] + 1)
            self.faces_data[f'face{action}']['receivedData'] += 1
            self.faces_data[f'face{action}']['MovingReceivedData'] += 1

            if self.reward_fct == 'simple':
                reward = 1 / normalized_rtt
            elif self.reward_fct == 'streak':
                reward = 1 / normalized_rtt + 0.1 * self.success_streak

        elif response_type == 'Nack':
            self.failure_streak += 1
            self.success_streak = 0
            self.faces_data[f'face{action}']['receivedNack'] += 1

            if self.reward_fct == 'simple':
                reward = NACK_NEGATIVE_REWARD
            elif self.reward_fct == 'streak':
                # reward = NACK_NEGATIVE_REWARD - 0.1 * math.exp(0.05 * self.failure_streak)
                reward = NACK_NEGATIVE_REWARD - 0.1 * self.failure_streak

        elif response_type == 'Timeout':
            self.failure_streak += 1
            self.success_streak = 0
            self.faces_data[f'face{action}']['nbTimeout'] += 1

            if self.reward_fct == 'simple':
                reward = TIMEOUT_NEGATIVE_REWARD
            elif self.reward_fct == 'streak':
                # reward = NACK_NEGATIVE_REWARD - 0.1 * math.exp(0.05 * self.failure_streak)
                reward = TIMEOUT_NEGATIVE_REWARD - 0.1 * self.failure_streak

        return reward

    def scenario_dynamic(self):
        # Scenario-specific dynamics
        if self.scenario == "static_with_one_optimal_face":
            # In this scenario, there are no dynamic changes, pass directly
            pass

        elif self.scenario == "dynamic_RTT":
            # Simulate dynamic RTT changes
            self._simulate_dynamic_rtt()

        elif self.scenario == "permanent_face_specific":
            # Apply permanent changes to specific faces.
            self._simulate_permanent_face_specific()

        elif self.scenario == "transient_face_specific":
            # Apply and then revert changes to face parameters
            self._simulate_transient_face_specific()

        elif self.scenario == "permanent_faulty_face":
            # Set certain faces as permanently faulty
            self._simulate_permanent_faulty_faces()

        elif self.scenario == "transient_faulty_face":
            # Temporary faults in faces, need to handle the duration and reversion logic
            self._simulate_transient_faulty_faces()

        elif self.scenario == "streak_disruption":
            # Simulate streak disruption
            self._simulate_streak_disruption()

        else:
            print(f"Unknown scenario: {self.scenario}")
            print(f"Please check the scenario name. \nAvailable scenarios are: {', '.join(self.available_scenarios)}")
            sys.exit(1)

    def step(self, action):
        """
        Executes one step of the environment's dynamics.

        Args:
            action (int): The action to be taken by the agent.

        Returns:
            tuple: A tuple containing the following:
                - observation (np.ndarray): Agent's observation of the current environment.
                - reward (float): Reward received by the agent after taking the action.
                - done (bool): Whether the episode has ended.
                - info (dict): Optional dictionary containing additional information about the step.
        """

        self.scenario_dynamic()

        # Simulate router behavior based on the chosen action for the current face
        # Retrieve the selected face based on action
        self.selected_face = self._get_face(action)

        # update selected face stats
        self.faces_data[self.selected_face]['sentInterests'] += 1

        if self.current_step % MOVING_WINDOW_SIZE == 0:
            for face_key in self.faces_data:
                self.faces_data[face_key]['MovingSentInterests'] = 0
                self.faces_data[face_key]['MovingReceivedData'] = 0

            for face_key in self.faces_data:
                for current_prefix in self.interests:
                    self.faces_data[face_key]['prefixAttempts'][current_prefix] = 0
                    self.faces_data[face_key]['prefixSuccesses'][current_prefix] = 0

        self.faces_data[self.selected_face]['MovingSentInterests'] += 1

        self.faces_data[self.selected_face]['prefixAttempts'][self.current_prefix] += 1

        allowed_prefixes = self.face_prefix_sensitivity[self.selected_face]

        if self.current_prefix in allowed_prefixes:
            # Proceed with regular handling if the prefix matches the face's sensitivity
            response_type = self._determine_response_type(self.selected_face)
        else:
            # Generate a Nack if the current prefix is not handled by the selected face
            response_type = 'Nack'

        # Process the selected response type and calculate the reward
        reward = self._process_response(action, response_type)

        # update router state
        selected_face_data = self.faces_data[self.selected_face]
        # state one includes raw stats
        if self.state == 'state_one':
            self.router_state[self.selected_face]['sentInterests'] = selected_face_data['sentInterests']
            self.router_state[self.selected_face]['receivedData'] = selected_face_data['receivedData']
            self.router_state[self.selected_face]['lastRTT'] = selected_face_data['lastRTT']
            self.router_state[self.selected_face]['avgRTT'] = selected_face_data['avgRTT']
            self.router_state[self.selected_face]['nbTimeout'] = selected_face_data['nbTimeout']
            self.router_state[self.selected_face]['receivedNack'] = selected_face_data['receivedNack']

        # state two includes aggregated states: success_rate and moving_avg_rtt
        elif self.state == 'state_two':
            # Calculate the success rate for the selected face
            sent_interests = selected_face_data['sentInterests']
            received_data = selected_face_data['receivedData']
            success_rate = received_data / sent_interests if sent_interests > 0 else 0.0
            self.router_state[self.selected_face]['successRate'] = success_rate

            # Calculate the overall moving average RTT for the selected face across all prefixes
            total_avg_rtt = 0.0
            count = 0
            for prefix, rtt_deque in selected_face_data['movingAvgRTTPrefix'].items():
                if len(rtt_deque) > 0:
                    prefix_avg_rtt = sum(rtt_deque) / len(rtt_deque)
                    total_avg_rtt += prefix_avg_rtt
                    count += 1
            moving_avg_rtt = total_avg_rtt / count if count > 0 else 0.0
            self.router_state[self.selected_face]['movingAvgRTT'] = moving_avg_rtt

        elif self.state == 'state_three':
            # Calculate the moving success rate for the selected face
            sent_interests = selected_face_data['MovingSentInterests']
            received_data = selected_face_data['MovingReceivedData']
            success_rate = received_data / sent_interests if sent_interests > 0 else 0.0
            self.router_state[self.selected_face]['movingSuccessRate'] = success_rate

            # Calculate the overall moving average RTT for the selected face across all prefixes
            total_avg_rtt = 0.0
            count = 0
            for prefix, rtt_deque in selected_face_data['movingAvgRTTPrefix'].items():
                if len(rtt_deque) > 0:
                    prefix_avg_rtt = sum(rtt_deque) / len(rtt_deque)
                    total_avg_rtt += prefix_avg_rtt
                    count += 1
            moving_avg_rtt = total_avg_rtt / count if count > 0 else 0.0
            self.router_state[self.selected_face]['movingAvgRTT'] = moving_avg_rtt

        elif self.state == 'state_four':
            selected_face_data = self.faces_data[self.selected_face]
            current_prefix = self.current_prefix

            # Calculate the moving success rate for the selected face
            sent_interests = selected_face_data['MovingSentInterests']
            received_data = selected_face_data['MovingReceivedData']
            success_rate = received_data / sent_interests if sent_interests > 0 else 0.0
            self.router_state[self.selected_face]['movingSuccessRate'] = success_rate

            total_avg_rtt = 0
            count = 0
            for prefix, rtt_deque in selected_face_data['movingAvgRTTPrefix'].items():
                if len(rtt_deque) > 0:
                    prefix_avg_rtt = sum(rtt_deque) / len(rtt_deque)
                    total_avg_rtt += prefix_avg_rtt
                    count += 1
            overall_moving_avg_rtt = total_avg_rtt / count if count > 0 else 0.0

            self.router_state[self.selected_face]['movingAvgRTT'] = overall_moving_avg_rtt

            # Calculate the success rate for the current prefix
            attempts = selected_face_data['prefixAttempts'][current_prefix]
            successes = selected_face_data['prefixSuccesses'][current_prefix]
            if attempts > 0:
                success_rate = successes / attempts
            else:
                success_rate = 0.0  # No attempts made, thus no success rate

            self.router_state[f'face{action}']['movingSuccessRatePerPrefix'] = success_rate

            # Calculate the moving average RTT for the current prefix
            rtt_deque = selected_face_data['movingAvgRTTPrefix'][current_prefix]
            if rtt_deque:
                moving_avg_rtt_per_current_prefix = sum(rtt_deque) / len(rtt_deque)
            else:
                moving_avg_rtt_per_current_prefix = 0.0  # No RTT data available, thus no average RTT

            self.router_state[f'face{action}']['movingAvgRTTPerPrefix'] = moving_avg_rtt_per_current_prefix

        # generate the next interest
        self.current_prefix = self.local_random.choice(self.interests)

        # Update current step count
        self.current_step += 1

        # if self.render_mode == "human":
        #     self._render_human()

        # Check if the maximum episode length is reached
        truncated = self.current_step >= self.max_episode_length

        # Reset the environment if the maximum episode length is reached
        if truncated:
            self.reset()

        # Return observation, reward, done, truncated, and additional information
        return self._get_obs(), reward, False, truncated, self._get_info()

    def _get_face(self, action):
        """
        Retrieves the face corresponding to the given action.

        Args:
            action (int): Action index representing the selected face.

        Returns:
            str: Name of the selected face.
        """
        faces = list(self.router_state.keys())
        return faces[action]

    def _calculate_reward(self):
        """
        Calculates the reward based on the router's state.

        Returns:
            float: Reward computed based on the router's state.

        """
        # Calculate reward based on the router's state
        total_data_received = sum(self.router_state[face]['receivedData'] for face in self.router_state)
        total_nack_received = sum(self.router_state[face]['receivedNack'] for face in self.router_state)
        total_timeout = sum(self.router_state[face]['nbTimeout'] for face in self.router_state)

        reward = total_data_received - total_nack_received - total_timeout

        return reward

    # ########################################################################################
    # Functions related to dynamic conditions
    # ########################################################################################
    def _simulate_dynamic_rtt(self):
        # Simulate changing RTT means dynamically during the simulation
        if self.current_step == 500:
            self.means = abs(self.local_random_np.normal(2 * MEAN_RTT_FACES, 2 * STANDARD_DEVIATION_MEAN_RTT_FACES, self.num_faces))
            self.max_RTTs = np.ceil(self.means + 3 * self.stDev)

    def _simulate_permanent_face_specific(self):
        if self.current_step == 0:
            # Map each face to a specific prefix (e.g., face i is sensitive to prefix i)
            self.face_prefix_sensitivity = {f'face{i}': [i] for i in range(self.num_faces)}

    def _simulate_transient_face_specific(self):
        start_step = 0  # Define the step at which changes begin
        duration = 500  # Duration of the changes in steps
        end_step = start_step + duration  # Calculate when to revert the changes

        # Apply changes at the start_step
        if self.current_step == start_step:
            # Temporarily change prefix sensitivity for specific faces
            self.face_prefix_sensitivity = {f'face{i}': list(self.interests) for i in range(self.num_faces)}
        # Revert changes after the duration ends
        if self.current_step == end_step:
            self.face_prefix_sensitivity = {f'face{i}': [i] for i in range(self.num_faces)}

    def _simulate_permanent_faulty_faces(self):
        # Set high nack and timeout probabilities for the best face
        if self.current_step == 500:
            best_face = self._get_face(np.argmin(self.means))
            self.nack_probability[best_face] = 0.5
            self.timeout_probability[best_face] = 0.5

    def _simulate_transient_faulty_faces(self):
        best_face = self._get_face(np.argmin(self.means))

        if self.current_step == 300:
            self.nack_probability[best_face] = 0.5
            self.timeout_probability[best_face] = 0.5

        if self.current_step == 600:
            self.nack_probability[best_face] = DEFAULT_NACK_PROBABILITY
            self.timeout_probability[best_face] = DEFAULT_TIMEOUT_PROBABILITY

    def _simulate_streak_disruption(self):
        if self.current_step % 100 == 0 and self.current_step != 0:
            # print(f"Current simulation step: {self.current_step}")
            face_key = f'face{self._best_face_index}'
            # print("optimal_face_index", self._best_face_index)
            self.nack_probability[face_key] = 1 - self.nack_probability[face_key]

    # ##############################################################################################################################
    # Changing Network Conditions: Simulate scenarios where network conditions change over time.
    # For example, varying RTT means or introducing sudden increases in network traffic.

    # Simulate congestion on the best face
    def _simulate_face_congestion(self):
        # Update face-specific parameters dynamically during the simulation
        if self.current_step == 500:
            # Change the timeout_probability for best face
            best_face = self._get_face(np.argmin(self.means))
            self.timeout_probability[best_face] = 0.5

    # Change the nack_probability for best face
    def _update_face_nack(self):
        # Update face-specific parameters dynamically during the simulation
        if self.current_step == 500:
            # Change the nack_probability for best face
            best_face = self._get_face(np.argmin(self.means))
            self.nack_probability[best_face] = 1.0
        elif self.current_step == 600:
            # Change the nack_probability for best face
            best_face = self._get_face(np.argmin(self.means))
            self.nack_probability[best_face] = 0.0

    def _update_probabilities(self):
        # Update probabilities dynamically during the simulation
        # For illustration, we'll just update them randomly
        for face in self.nack_probability:
            self.nack_probability[face] = self.local_random_np.uniform(0.01, 0.05)
            self.timeout_probability[face] = self.local_random_np.uniform(0.01, 0.05)

    # Update Parameters Dynamically: Modify parameters such as RTT means, probabilities of generating Nack or Timeout
    def _update_parameters(self):
        # Update parameters dynamically during the simulation
        # For example, you can change RTT means or probabilities based on the current step
        if self.current_step % 100 == 0:
            self.means = abs(self.local_random_np.normal(120, 30, self.num_faces))
            self._update_probabilities()

    # Change Number of Faces: Introduce variations in the number of faces dynamically during the simulation.
    def _update_num_faces(self):
        # Update the number of faces dynamically during the simulation
        if self.current_step == 500:
            self.num_faces = self.num_faces + 1  # Change to the desired number of faces
            self.reset()

    # Modify Face-specific Parameters: Change parameters specific to each face independently during the simulation.
    def _update_face_parameters(self):
        # Update face-specific parameters dynamically during the simulation
        if self.current_step == 200:
            # Change the nack_probability for face 0
            self.nack_probability['face0'] = 0.1

    # Introduce External Events: Simulate external events or disturbances that affect the router's behavior.
    def _simulate_external_events(self):
        # Simulate external events dynamically during the simulation
        if self.current_step == 300:
            # Trigger an external event that affects router behavior
            self._update_parameters()
            self._update_num_faces()

    # Adversarial Interference: Introduce adversarial interference by temporarily increasing
    # the probabilities of Nack and Timeout for certain faces.
    def _simulate_adversarial_interference(self):
        # Simulate adversarial interference dynamically during the simulation
        if 200 <= self.current_step <= 400:
            adversarial_face = 'face1'  # Choose a face for interference
            self.nack_probability[adversarial_face] = min(0.5, self.nack_probability[adversarial_face] + 0.02)
            self.timeout_probability[adversarial_face] = min(0.5, self.timeout_probability[adversarial_face] + 0.02)

    # Variable Number of Prefixes: Change the number of available prefixes dynamically during the simulation.
    def _simulate_variable_prefixes(self):
        # Simulate variable number of prefixes dynamically during the simulation
        if self.current_step == 500:
            self.interests = np.arange(0, self.num_faces * 2, 1)  # Change to the desired number of prefixes
            self.reset()

    # Learning Adversarial Strategies:Implement a scenario where the environment's behavior adapts
    # based on the agent's past actions, simulating an adversarial environment.
    def _simulate_adversarial_strategies(self, last_action):
        # Simulate learning adversarial strategies dynamically during the simulation
        if self.current_step > 300 and self.current_step % 50 == 0:
            adversarial_face = self._get_face(last_action)
            self.nack_probability[adversarial_face] = min(0.5, self.nack_probability[adversarial_face] + 0.02)
            self.timeout_probability[adversarial_face] = min(0.5, self.timeout_probability[adversarial_face] + 0.02)

    # Learning Rivalry: Implement a scenario where the environment learns the agent's preferences
    # and adversarially adjusts the RTT for non-selected faces.
    def _simulate_learning_rivalry(self, last_action):
        # Simulate learning rivalry dynamically during the simulation
        if self.current_step > 200 and self.current_step % 80 == 0:
            non_selected_faces = [f for f in self.router_state if f != self._get_face(last_action)]
            for face in non_selected_faces:
                index_from_face = int(face[4:])
                self.means[index_from_face] += 10

    # Multi-Stage Environment with Time-Varying RTT: Implement an environment with multiple stages,
    # each having different RTT distributions, simulating a network with varying characteristics over time.
    def _simulate_multi_stage_rtt(self):
        # Define different RTT distributions for each stage
        stages_rtt = [
            self.local_random_np.normal(100, 20, self.num_faces),
            self.local_random_np.normal(150, 30, self.num_faces),
            self.local_random_np.normal(80, 15, self.num_faces)
        ]

        # Determine the current stage based on the step count
        current_stage = min(self.current_step // 300, len(stages_rtt) - 1)

        # Update RTT means dynamically during the simulation
        self.means = abs(stages_rtt[current_stage])

    # Bandwidth Fluctuations: Simulate fluctuations in available bandwidth,
    # affecting the RTT and data transmission rates.
    def _simulate_bandwidth_fluctuations(self):
        # Simulate bandwidth fluctuations dynamically during the simulation
        if self.current_step % 200 == 0:
            self.stDev = max(0.5, self.stDev - 0.1)

    # Adaptive Prefix Popularity: Change the popularity of prefixes over time, simulating shifts in content demand.
    def _simulate_adaptive_prefix_popularity(self):
        # Simulate adaptive prefix popularity dynamically during the simulation
        if self.current_step % 150 == 0:
            prefix_weights = self.local_random.uniform(0.5, 1.5, self.num_faces)
            self.interests = self.local_random.choice(np.arange(0, self.num_faces), size=self.num_faces, p=prefix_weights)
            self.reset()

    # Temporal Face Sensitivity: Introduce temporal variations in faces sensitivity to prefixes.
    # TODO this could be done by face, e.g., self.face_prefix_sensitivity[face] = not self.face_prefix_sensitivity[face]
    def _simulate_temporal_face_sensitivity(self):
        # Simulate temporal face sensitivity variations dynamically during the simulation
        if self.current_step % 100 == 0:
            self.face_prefix_sensitivity = not self.face_prefix_sensitivity

    # Network Congestion and Adaptive Timeout: Simulate network congestion affecting the timeout probabilities,
    # requiring agents to adapt their strategies.
    def _simulate_network_congestion(self):
        # Simulate network congestion dynamically during the simulation
        if self.current_step % 150 == 0:
            congestion_factor = self.local_random.uniform(0.5, 1.5)
            for face in self.timeout_probability:
                self.timeout_probability[face] *= congestion_factor
                self.timeout_probability[face] = min(self.timeout_probability[face], 0.2)

    # Functions related to the rendring
    def render(self):
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'rgb_array':
            self._render_rgb()
        else:
            super(NDNRouterEnv, self).render()

    def _render_human(self):
        # Print the current timestep
        print(f"Current timestep: {self.current_step}")

        # Print the state of each face
        for face, face_state in self.faces_data.items():
            face_info = (
                f"Face: {face}, "
                f"Sent Interests: {face_state['sentInterests']}, "
                f"Received Data: {face_state['receivedData']}, "
                f"Received Nack: {face_state['receivedNack']}, "
                f"Number of Timeouts: {face_state['nbTimeout']}, "
                f"Average RTT: {face_state['avgRTT']}, "
                f"Last RTT: {face_state['lastRTT']}"
            )
            print(face_info)

        # Print router's state
        total_interest_sent = sum(self.faces_data[face]['sentInterests'] for face in self.router_state)
        total_data_received = sum(self.faces_data[face]['receivedData'] for face in self.router_state)
        router_info = (
            f"Router's state: "
            f"Total Sent Interests: {total_interest_sent}, "
            f"Total Received Data: {total_data_received}"
        )
        print(router_info)

    def _render_rgb(self):
        # faces = list(self.router_state.keys())
        sent_interests = [face_state['sentInterests'] for face_state in self.router_state.values()]
        np_array = np.array(sent_interests)
        return np_array

    def close(self):
        pass
