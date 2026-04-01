import numpy as np
import pickle


class EGreedyMultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.q_values = np.zeros(num_arms)  # Initialize the estimated values of each arm
        self.action_counts = np.zeros(num_arms)  # Track the number of times each arm has been pulled

    def select_arm(self):
        # Explore with epsilon-greedy strategy
        epsilon = 0.1
        if np.random.rand() < epsilon:
            # Explore: choose a random arm
            return np.random.choice(self.num_arms)
        else:
            # Exploit: choose the arm with the highest estimated value
            return np.argmax(self.q_values)

    def update_arm(self, arm, reward):
        # Update the estimated value of the chosen arm based on the observed reward
        self.action_counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.action_counts[arm]

    def get_q_values(self):
        # Return the estimated values of each arm
        return self.q_values


class UCBMultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.q_values = np.zeros(num_arms)  # Initialize the estimated values of each arm
        self.action_counts = np.zeros(num_arms)  # Track the number of times each arm has been pulled
        self.total_pulls = 0  # Track the total number of arm pulls

    def select_arm(self):
        # UCB exploration strategy
        c = 2.0  # You can adjust this exploration weight parameter
        ucb_values = self.q_values + c * np.sqrt(np.log(self.total_pulls + 1) / (self.action_counts + 1e-6))
        return np.argmax(ucb_values)

    def update_arm(self, arm, reward):
        # Update the estimated value of the chosen arm based on the observed reward
        self.action_counts[arm] += 1
        self.total_pulls += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.action_counts[arm]

    def get_q_values(self):
        # Return the estimated values of each arm
        return self.q_values


class BetaThompsonSamplingMultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms)  # Initialize the alpha parameter for each arm (for Beta distribution)
        self.beta = np.ones(num_arms)  # Initialize the beta parameter for each arm (for Beta distribution)

    def select_arm(self):
        # Sample from the Beta distribution for each arm
        sampled_values = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_values)

    def update_arm(self, arm, reward):
        # Update the alpha and beta parameters based on the observed reward
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def get_sampled_parameters(self):
        # Return the sampled parameters (alpha and beta) for each arm
        return self.alpha, self.beta


# class GaussianThompsonSamplingMultiArmedBandit:
#     def __init__(self, num_arms, prior_mean=0, prior_std=1):
#         self.num_arms = num_arms
#         self.prior_mean = prior_mean
#         self.prior_std = prior_std
#
#         # Initialize the prior distribution for each arm
#         self.arm_means = np.full(num_arms, prior_mean)
#         self.arm_stds = np.full(num_arms, prior_std)
#
#     def select_arm(self):
#         # Sample from the posterior distribution for each arm
#         samples = np.random.normal(self.arm_means, self.arm_stds)
#         return np.argmax(samples)
#
#     def update_arm(self, arm, reward):
#         # Update the posterior distribution for the chosen arm based on the observed reward
#         # Using Bayesian update with Gaussian prior and likelihood
#         epsilon = 1e-8  # Small positive constant to avoid division by zero
#
#         posterior_std = np.sqrt(1 / ((1 / self.prior_std ** 2) + (1 / (self.arm_stds[arm] ** 2 + epsilon))))
#         posterior_mean = ((self.prior_mean / self.prior_std ** 2) + (reward / (self.arm_stds[arm] ** 2 + epsilon))) * (
#                 1 / ((1 / self.prior_std ** 2) + (1 / (self.arm_stds[arm] ** 2 + epsilon))))
#
#         # Update the parameters of the posterior distribution for the chosen arm
#         self.arm_means[arm] = posterior_mean
#         self.arm_stds[arm] = posterior_std
#
#     def get_posterior_params(self):
#         # Return the parameters of the posterior distribution for each arm
#         return self.arm_means, self.arm_stds


class SoftmaxMultiArmedBandit:
    def __init__(self, num_arms, tau=0.1):
        self.num_arms = num_arms
        self.tau = tau
        self.arm_values = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)

    def select_arm(self):
        probabilities = self._compute_softmax_probabilities()
        chosen_arm = np.random.choice(self.num_arms, p=probabilities)
        return chosen_arm

    def update_arm(self, arm, reward):
        self.arm_counts[arm] += 1
        self.arm_values[arm] += (reward - self.arm_values[arm]) / self.arm_counts[arm]

    def _compute_softmax_probabilities(self):
        exp_values = np.exp(self.arm_values / self.tau)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities


class UnknownMeanUnknownVariance:
    def __init__(self):
        self.n = 0  # the number of times this arm has been tried

        self.α = 1  # gamma shape parameter
        self.β = 1  # gamma rate parameter

        self.μ_0 = 0  # the prior (estimated) mean
        self.v_0 = self.β / (self.α + 1)  # the prior (estimated) variance

    def update(self, x):
        ''' increase the number of times this arm has been used and improve the estimate of the
            mean and variance by combining the single new value 'x=reward' with the current estimate '''
        n = 1
        v = self.n

        self.α = self.α + n / 2
        self.β = self.β + ((n * v / (v + n)) * (((x - self.μ_0) ** 2) / 2))

        # estimate the variance - calculate the mean from the gamma hyperparameters
        self.v_0 = self.β / (self.α + 1)

        self.n += 1
        self.μ_0 = self.μ_0 + (x - self.μ_0) / self.n
        # self.μ_0 = np.array(self.x).mean()

    def sample(self):
        ''' sample from our estimated normal '''

        precision = np.random.gamma(self.α, 1 / self.β)
        if precision == 0 or self.n == 0: precision = 0.001

        estimated_variance = 1 / precision
        return np.random.normal(self.μ_0, np.sqrt(estimated_variance))


# return the index of the largest value in the supplied list
# - arbitrarily select between the largest values in the case of a tie
# (the standard np.argmax just chooses the first value in the case of a tie)
def random_argmax(value_list):
    """ a random tie-breaking argmax"""
    values = np.asarray(value_list)
    return np.argmax(np.random.random(values.shape) * (values == values.max()))


class GaussianThompsonSampling:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arms = [UnknownMeanUnknownVariance() for _ in range(num_arms)]

    def select_arm(self):
        # choose the arm with the current highest sampled value or
        # arbitrary select an arm in the case of a tie
        arm_samples = [arm.sample() for arm in self.arms]
        arm_index = random_argmax(arm_samples)
        return arm_index

    def update_arm(self, arm_index, reward):
        self.arms[arm_index].update(reward)

    # class ContextualMultiArmedBandit:


#     def __init__(self, num_faces):
#         self.num_faces = num_faces
#         self.num_features = num_faces * 6 + 1  # Number of features in the observation space
#         self.weights = np.zeros((self.num_faces, self.num_features))
#         # self.theta = np.random.randn(self.num_faces, self.num_features)
#
#     def select_arm(self, context):
#         probabilities = expit(np.dot(self.weights, context))
#         chosen_arm = np.argmax(probabilities)
#         return chosen_arm
#
#     def update_arm(self, arm, reward, context):
#         self.weights[arm] += reward * context  # Update the weights based on the observed reward


# class LinUCB:
#     def __init__(self, num_faces, num_features, alpha=1.0):
#         self.num_faces = num_faces
#         self.num_features = num_features  # Number of features in the observation space
#         self.alpha = alpha
#
#         # Initialize parameters for each arm
#         self.A = [np.identity(self.num_features) for _ in range(self.num_faces)]
#         self.b = [np.zeros((self.num_features, 1)) for _ in range(self.num_faces)]
#
#     def select_arm(self, context):
#         estimated_rewards = [np.dot(self.A[a], context).T @ np.linalg.inv(self.A[a]) @ self.b[a] +
#                              self.alpha * np.sqrt(context.T @ np.linalg.inv(self.A[a]) @ context)
#                              for a in range(self.num_faces)]
#
#         return np.argmax(estimated_rewards)
#
#     def update_arm(self, arm, context, reward):
#         self.A[arm] += np.outer(context, context)
#         self.b[arm] += reward * context.reshape((-1, 1))


class LinUCBAgent:
    def __init__(self, num_arms, context_dim, alpha=1.0):
        self.num_arms = num_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = {arm: np.identity(context_dim) for arm in range(num_arms)}
        self.b = {arm: np.zeros((context_dim, 1)) for arm in range(num_arms)}
        # self.model = {arm: Ridge(alpha=self.alpha) for arm in range(num_arms)}

    def select_arm(self, context):
        p_values = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            theta_hat = np.linalg.inv(self.A[arm]).dot(self.b[arm])
            uncertainty = np.sqrt(context.T.dot(np.linalg.inv(self.A[arm])).dot(context))
            p_values[arm] = theta_hat.T.dot(context) + self.alpha * uncertainty

        chosen_arm = np.argmax(p_values)
        return chosen_arm

    def update_arm(self, arm, context, reward):
        self.A[arm] += context.dot(context.T)
        self.b[arm] += reward * np.expand_dims(context, axis=1)
        # self.model[arm].fit(context.T.reshape(1, -1), [reward])


class LinUCB:
    def __init__(self, n_actions, n_features, alpha=1.0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters
        self.A = np.array(
            [np.identity(n_features) for _ in range(n_actions)]
        )  # action covariance matrix
        self.b = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )  # action reward vector
        self.theta = np.array(
            [np.zeros(n_features) for _ in range(n_actions)]
        )  # action parameter vector

    def select_arm(self, context):
        context = np.array(context)  # Convert list to ndarray
        context = context.reshape(
            -1, 1
        )  # reshape the context to a single-column matrix
        p = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            theta = np.dot(
                np.linalg.inv(self.A[a]), self.b[a]
            )  # theta_a = A_a^-1 * b_a
            theta = theta.reshape(-1, 1)  # Explicitly reshape theta
            p[a] = np.dot(theta.T, context) + self.alpha * np.sqrt(
                np.dot(context.T, np.dot(np.linalg.inv(self.A[a]), context))
            )  # p_t(a|x_t) = theta_a^T * x_t + alpha * sqrt(x_t^T * A_a^-1 * x_t)

        chosen_arm = np.argmax(p)
        return chosen_arm

    def update_arm(self, action, context, reward):
        self.A[action] += np.outer(context, context)  # A_a = A_a + x_t * x_t^T
        self.b[action] += reward * context  # b_a = b_a + r_t * x_tx

# class GaussianThompsonSampling:
#     def __init__(self, num_arms, prior_mean=0, prior_std=1):
#         self.num_arms = num_arms
#
#         self.prior_mean = prior_mean
#         self.prior_std = prior_std
#
#         # Initialize the prior distribution for each arm
#         self.arm_means = np.full(num_arms, prior_mean)
#         self.arm_stds = np.full(num_arms, prior_std)
#
#     def select_arm(self):
#         # Sample from the posterior distribution for each arm
#         samples = np.random.normal(self.arm_means, self.arm_stds)
#         return np.argmax(samples)
#
#     def update_arm(self, arm, reward):
#         # Update the posterior distribution for the chosen arm based on the observed reward
#         # Using Bayesian update with Gaussian prior and likelihood
#         epsilon = 1e-8  # Small positive constant to avoid division by zero
#
#         posterior_std = np.sqrt(1 / ((1 / self.prior_std ** 2) + (1 / (self.arm_stds[arm] ** 2 + epsilon))))
#         posterior_mean = ((self.prior_mean / self.prior_std ** 2) + (reward / (self.arm_stds[arm] ** 2 + epsilon))) * (
#                 1 / ((1 / self.prior_std ** 2) + (1 / (self.arm_stds[arm] ** 2 + epsilon))))
#
#         # Update the parameters of the posterior distribution for the chosen arm
#         self.arm_means[arm] = posterior_mean
#         self.arm_stds[arm] = posterior_std
#
#     def get_posterior_params(self):
#         # Return the parameters of the posterior distribution for each arm
#         return self.arm_means, self.arm_stds
