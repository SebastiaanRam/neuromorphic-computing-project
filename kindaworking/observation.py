import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# Environment setup
env = gym.make("LunarLander-v2")
input_neurons = env.observation_space.shape[0]  # m = 8
motor_neurons = env.action_space.n             # n_total = 4

# Hyperparameters
learning_rate = 0.005

# Add exploration decay
exploration_level_max = 1
exploration_level_min = 0.1
decay_rate = 0.001  # Decay rate
use_exploration_decay = True
exploration_level = exploration_level_max
num_episodes = 5000
max_steps = 1000

# Initialize weights
weights = np.random.uniform(-0.1, 0.1, (motor_neurons, input_neurons))

def get_synaptic_input(weights, inputs, variance):
    noise = np.random.normal(0, variance, weights.shape[0])
    return np.dot(weights, inputs) + noise

def activation_function(inputs):
    return np.maximum(0, inputs)  # ReLU activation

def low_pass_filter(prev, curr, alpha=0.8):
    return (1 - alpha) * prev + alpha * curr

def normalize_weights(weights, max_norm=1.0):
    """
    Normalize weights to prevent unbounded growth.
    """
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8) 
    norms = np.clip(norms, a_min=None, a_max=max_norm)
    return weights / norms

# Training
rewards = []
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
        # reward = np.clip(reward, -50, 50)  # Normalize reward

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
    
cumsum_rewards = np.cumsum(rewards)
plt.plot(cumsum_rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("Using observations")
plt.grid()
plt.savefig("kindaworking/observation_cumsum.png")


# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Using observations")
plt.legend()
plt.grid()
plt.savefig("kindaworking/observation_rewards.png")