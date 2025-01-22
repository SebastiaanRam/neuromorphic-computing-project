import gymnasium as gym
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use a backend suitable for your environment
import matplotlib.pyplot as plt


# Functions remain similar with slight improvements where necessary

def get_s_neuron_activity(a_input):
    return np.maximum(0, a_input)


def get_a_synaptic_input(w, x):
    return np.dot(w, x)


def get_x_neuron_activity(y_star, weights, n_input, d):
    y_star = y_star / np.linalg.norm(y_star)
    Q = np.random.uniform(-1, 1, (d, n_input))
    Q_inverse = np.linalg.pinv(Q)
    s_tilde = np.dot(Q_inverse, y_star)
    x_activities = np.dot(np.linalg.pinv(weights), s_tilde)
    return x_activities


def delta(x):
    return max(0, x)


def get_cursor_velocity(s_activities, p_directions, k_s=0.03, alpha=1, beta=0, d=2):
    normalized_activities = (s_activities - beta) / alpha
    population_vector = np.sum(
        normalized_activities[:, np.newaxis] * p_directions, axis=0
    )
    y_t = k_s * (d / len(s_activities)) * population_vector
    return y_t * 100


def r_ang(y, y_star, reward):
    return reward + np.dot(y, y_star) * np.linalg.norm(y)


def low_pass_filter(previous, current, alpha=0.8):
    return (1 - alpha) * previous + alpha * current


# Initialize environment and parameters
env = gym.make("LunarLander-v2")
n_input = env.observation_space.shape[0]  # Number of state features
n_total = 8  # Number of neurons
d = 2  # Movement dimensionality
weights = np.random.rand(n_input, n_total) * 0.01
preferred_directions = np.random.uniform(-1, 1, (n_total, d))
preferred_directions = preferred_directions / np.linalg.norm(preferred_directions, axis=1, keepdims=True)

learning_rate = 0.01
learning_rate_decay = 0.999
variance = 0.1

total_rewards = []
n_episodes = 500

for i in range(n_episodes):
    state, info = env.reset()
    total_reward = 0
    R_t = 0
    old_a_input = 0
    done = False

    while not done:
        # Normalize state to fit neuron inputs
        normalized_state = state / np.linalg.norm(state)
        x_activities = get_x_neuron_activity(normalized_state, weights, n_input, d)
        a_input = get_a_synaptic_input(weights, x_activities)
        s_activities = get_s_neuron_activity(a_input)

        # Use activities to compute cursor velocity
        y_t = get_cursor_velocity(s_activities, preferred_directions)

        # Select action based on activities
        action = np.argmax(s_activities[:env.action_space.n])

        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Compute desired direction
        y_star = np.array([state[2], state[3]])  # Target velocity
        y_star = y_star / np.linalg.norm(y_star) if np.linalg.norm(y_star) > 0 else np.zeros_like(y_star)

        # Update reward estimate
        new_R_t = r_ang(y_t, y_star, reward)
        R_hat = low_pass_filter(R_t, new_R_t)
        R_t = new_R_t

        a_input_hat = low_pass_filter(old_a_input, a_input)
        old_a_input = a_input

        a_difference = a_input - a_input_hat
        r_difference = R_t - R_hat

        # Hebbian learning rule
        delta_weights = learning_rate * np.outer(x_activities, a_difference).T * r_difference
        weights += delta_weights

        # Decay learning rate
        learning_rate = max(0.001, learning_rate * learning_rate_decay)

        state = next_state

    total_rewards.append(total_reward)
    print(f"Episode {i + 1}/{n_episodes} - Reward: {total_reward}")

# Plot rewards
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards Over Episodes")
plt.grid()
plt.savefig("reward_plot.png")
env.close()
