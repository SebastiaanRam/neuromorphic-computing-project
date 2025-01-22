import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("LunarLander-v2")
n_actions = env.action_space.n  # Number of actions
n_observations = env.observation_space.shape[0]  # Observation space size

# Hyperparameters
hidden_neurons = 16  # Number of neurons in the hidden layer
learning_rate = 0.01
noise_level = 0.1
num_episodes = 5000
max_steps = 1000

# Initialize weights (input to hidden and hidden to output layers)
input_to_hidden_weights = np.random.uniform(-0.5, 0.5, (hidden_neurons, n_observations))
hidden_to_output_weights = np.random.uniform(-0.5, 0.5, (n_actions, hidden_neurons))

def compute_hidden_activities(state, weights):
    """Compute hidden layer activities using ReLU activation."""
    activities = np.dot(weights, state)
    return np.maximum(0, activities)  # ReLU activation

def compute_output_activities(hidden_activities, weights, noise_level):
    """Compute output layer activities with added noise."""
    noisy_input = np.dot(weights, hidden_activities) + np.random.normal(0, noise_level, size=weights.shape[0])
    return np.maximum(0, noisy_input)  # ReLU activation

def low_pass_filter(previous, current, alpha=0.8):
    """Low-pass filter for smoothing signals."""
    return (1 - alpha) * previous + alpha * current

# Training variables
episode_rewards = []
previous_reward = 0
previous_output_activities = np.zeros(n_actions)

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    previous_output_activities = np.zeros(n_actions)
    for step in range(max_steps):
        # Compute hidden and output layer activities
        hidden_activities = compute_hidden_activities(state, input_to_hidden_weights)
        output_activities = compute_output_activities(hidden_activities, hidden_to_output_weights, noise_level)

        # Select action based on output activities
        action = np.argmax(output_activities)

        # Step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Low-pass filter for activities and reward
        filtered_output_activities = low_pass_filter(previous_output_activities, output_activities)
        filtered_reward = low_pass_filter(previous_reward, reward)

        # Compute differences
        output_difference = output_activities - filtered_output_activities
        reward_difference = reward - filtered_reward

        # Reward-modulated Hebbian learning rule
        delta_hidden_to_output = learning_rate * np.outer(output_difference, hidden_activities) * reward_difference
        hidden_to_output_weights += delta_hidden_to_output

        # Update for the next step
        state = next_state
        previous_output_activities = output_activities
        previous_reward = reward

        if done:
            break

    # Store total reward for this episode
    episode_rewards.append(total_reward)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress with Reward-Modulated Hebbian Learning (CartPole-v1)")
plt.legend()
plt.grid()
plt.show()

env.close()
