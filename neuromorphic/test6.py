import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Environment setup
env = gym.make("LunarLander-v2")
actions = env.action_space.n
observations = env.observation_space.shape[0]

# Hyperparameters
m = observations
n_total = actions
hidden_neurons = 32
x_weights = np.random.uniform(-0.5, 0.5, (hidden_neurons, m))
a_weights = np.random.uniform(-0.5, 0.5, (n_total, hidden_neurons))
initial_learning_rate = 0.1
decay_rate = 0.001
exploration_variance_max = 0.5
exploration_variance_min = 0.1
max_episode_steps = 1000
num_episodes = 5000

# Helper functions
def get_a_synaptic_input(w, x, variance):
    noise = np.random.normal(0, variance, w.shape[0])
    return np.dot(w, x) + noise

def compute_x_activities(state, W_input, noise_level=0.01):
    x = np.dot(W_input, state)
    x = np.maximum(x, 0)
    noise = np.random.normal(0, noise_level, x.shape)
    return x / (np.linalg.norm(x) + 1e-8)

def low_pass_filter(previous, current, alpha=0.8):
    return (1 - alpha) * previous + alpha * current

def normalize_weights(weights, max_norm=1.0):
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    return weights / np.maximum(norms, 1e-8)

# Training
episode_rewards = []
for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    previous_a_input = np.zeros(n_total)
    previous_reward = 0
    exploration_variance = max(exploration_variance_min, exploration_variance_max * np.exp(-decay_rate * episode))
    learning_rate = initial_learning_rate / (1 + decay_rate * episode)
    for step in range(max_episode_steps):
        # Compute activities
        x_activities = compute_x_activities(observation, x_weights)
        current_a_input = get_a_synaptic_input(a_weights, x_activities, exploration_variance)
        motor_activities = np.maximum(0, current_a_input)
        action = np.argmax(motor_activities)

        # Step in environment
        next_observation, current_reward, done, _, _ = env.step(action)
        episode_reward += current_reward

        # Reward shaping
        shaped_reward = current_reward - 0.1 * np.linalg.norm(next_observation[2:4])
        if abs(next_observation[0]) < 0.1 and abs(next_observation[1]) < 0.1:
            shaped_reward += 10

        # Filtered values
        filtered_a_input = low_pass_filter(previous_a_input, current_a_input)
        filtered_reward = low_pass_filter(previous_reward, shaped_reward)

        # Weight update
        a_difference = (current_a_input - filtered_a_input) * 10
        r_difference = (shaped_reward - filtered_reward) * 10
        delta_weights = learning_rate * np.outer(a_difference, x_activities) * r_difference
        a_weights += delta_weights
        a_weights = normalize_weights(a_weights)

        observation = next_observation
        previous_a_input = current_a_input
        previous_reward = shaped_reward

        if done:
            break

    episode_rewards.append(episode_reward)

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Lunar Lander Training Progress")
plt.grid()
plt.savefig("lunar_lander_rewards.png")
env.close()
