import gymnasium as gym
import numpy as np

# Environment setup
env = gym.make("Pendulum-v1")

# Parameters
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
hidden_neurons = 100  # Number of motor cortical neurons
learning_rate = 1e-4  # Learning rate for weight updates
exploration_level = 0.1  # Exploration noise level
num_episodes = 500  # Number of training episodes

def activation(x):
    """Threshold-linear activation function."""
    return np.maximum(0, x)

# Initialize weights randomly (input to cortical neurons)
weights = np.random.uniform(-0.5, 0.5, (hidden_neurons, num_inputs))

# Define the reward-modulated Hebbian learning rule (EH rule)
def update_weights(weights, x, activation_i, activation_mean, reward, reward_mean):
    """Update weights according to the EH rule."""
    delta_weights = (
        learning_rate
        * np.outer(
            (activation_i - activation_mean),  # Postsynaptic deviation
            (x),  # Presynaptic activity
        )
        * (reward - reward_mean)  # Reward deviation
    )
    return weights + delta_weights

# Helper functions
def noisy_activity(inputs, weights, noise_level):
    """Calculate neuron activities with added noise."""
    total_input = np.dot(weights, inputs)
    noise = np.random.uniform(-noise_level, noise_level, total_input.shape)
    return activation(total_input + noise), total_input

# Training
reward_history = []

for episode in range(num_episodes):
    observation, _ = env.reset()
    cursor_position = np.zeros(num_inputs)  # Initialize cursor at origin
    target_position = np.random.uniform(-1, 1, size=num_inputs)  # Random target
    done = False
    total_reward = 0

    # Initialize averages
    reward_mean = 0
    activation_mean = np.zeros(hidden_neurons)

    while not done:
        # Step 1: Compute desired direction
        desired_direction = target_position - cursor_position
        desired_direction /= np.linalg.norm(desired_direction)  # Normalize

        # Step 2: Determine input activities
        input_activities = desired_direction

        # Step 3: Determine motor cortical activities with noise
        activities, raw_activities = noisy_activity(input_activities, weights, exploration_level)

        # Step 4: Determine new cursor location
        action = np.dot(activities, np.random.uniform(-1, 1, hidden_neurons))
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        next_observation, reward, terminated, truncated, _ = env.step([action])
        cursor_position += action * np.array([1, 0])  # Update cursor position for Pendulum (1D action)
        total_reward += reward

        # Step 5: Update synaptic weights using the EH rule
        weights = update_weights(weights, input_activities, raw_activities, activation_mean, reward, reward_mean)

        # Update running means (low-pass filtered)
        activation_mean = 0.8 * activation_mean + 0.2 * raw_activities
        reward_mean = 0.8 * reward_mean + 0.2 * reward

        # Step 6: Check if target is hit
        if np.linalg.norm(cursor_position - target_position) < 0.05:
            done = True

        # Move to next observation
        observation = next_observation

    # Store reward for analysis
    reward_history.append(total_reward)

    # Print progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

# Plot results
import matplotlib.pyplot as plt

plt.plot(reward_history)
plt.title("Training Progress")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
