import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("LunarLander-v2")
input_neurons = env.observation_space.shape[0]  # m = 8
motor_neurons = env.action_space.n             # n_total = 4

# Hyperparameters
learning_rate = 0.01
exploration_level_max = 1.0
exploration_level_min = 0.1
decay_rate = 0.001  # Decay rate for exploration decay
use_exploration_decay = False
exploration_level = 0.1
num_episodes = 5000
max_steps = 1000

# Initialize weights
weights = np.random.uniform(-0.5, 0.5, (motor_neurons, input_neurons))

def get_synaptic_input(weights, inputs, variance):
    """
    Compute synaptic input to motor cortex neurons.
    Includes noise for exploration.
    """
    noise = np.random.normal(0, variance, weights.shape[0])
    return np.dot(weights, inputs) + noise

def activation_function(inputs):
    """
    Activation function: ReLU to ensure non-negative activities.
    """
    return np.maximum(0, inputs)

def low_pass_filter(prev, curr, alpha=0.8):
    """
    Low-pass filter to smooth values for postsynaptic activity and reward.
    """
    return (1 - alpha) * prev + alpha * curr

def normalize_weights(weights, max_norm=1.0):
    """
    Normalize weights to prevent unbounded growth.
    """
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=None, a_max=max_norm)
    return weights / norms

# Training
rewards = []
angular_matches = []  # Track angular match as a metric

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    previous_a_input = np.zeros(motor_neurons)
    previous_reward = 0

    for step in range(max_steps):
        # Compute neuron activities
        a_input = get_synaptic_input(weights, obs, exploration_level)
        s_activities = activation_function(a_input)

        # Choose action based on max activity
        action = np.argmax(s_activities)

        # Step in the environment
        next_obs, reward, done, _, _ = env.step(action)
        episode_reward += reward

        # Reward-modulated Hebbian update
        filtered_a_input = low_pass_filter(previous_a_input, a_input)
        filtered_reward = low_pass_filter(previous_reward, reward)

        delta_weights = learning_rate * np.outer(
            (a_input - filtered_a_input),  # Postsynaptic deviation
            obs                          # Presynaptic activity
        ) * (reward - filtered_reward)  # Reward deviation

        weights += delta_weights
        weights = normalize_weights(weights)  # Normalize weights

        obs = next_obs
        previous_a_input = a_input
        previous_reward = reward

        if use_exploration_decay:
            # Decay exploration level
            exploration_level = exploration_level_min + (exploration_level_max - exploration_level_min) * np.exp(-decay_rate * episode)

        if done:
            break

    rewards.append(episode_reward)

# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Training with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("rewards_plot_improved.png")

# Plotting angular matches
plt.figure(figsize=(10, 6))
plt.plot(angular_matches, label="Angular Match")
plt.xlabel("Step")
plt.ylabel("Cosine Similarity")
plt.title("Angular Match Over Training Steps")
plt.legend()
plt.grid()
plt.savefig("angular_match_plot.png")

# Plot cumulative rewards
cumsum_rewards = np.cumsum(rewards)
plt.figure(figsize=(10, 6))
plt.plot(cumsum_rewards, label="Cumulative Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("cumsum_rewards_plot_improved.png")

env.close()
