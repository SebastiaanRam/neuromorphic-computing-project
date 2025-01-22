import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a backend suitable for your environment

# Environment setup
env = gym.make("Pendulum-v1")

# Parameters
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
hidden_neurons = 100  # Number of motor cortical neurons
learning_rate = 1e-4  # Learning rate for weight updates
exploration_level = 0.1  # Initial exploration noise level
num_episodes = 10000  # Number of training episodes
gamma = 0.99  # Discount factor for temporal credit assignment

# Activation function
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
    done = False
    total_reward = 0

    # Initialize averages
    reward_mean = 0
    activation_mean = np.zeros(hidden_neurons)
    step = 0

    while not done:
        # Step 1: Compute desired direction (normalized observation)
        input_activities = observation / np.linalg.norm(observation)

        # Step 2: Determine motor cortical activities with noise
        activities, raw_activities = noisy_activity(input_activities, weights, exploration_level)

        # Step 3: Convert activities to action (linear mapping for simplicity)
        action = np.tanh(np.dot(activities, np.random.uniform(-1, 1, hidden_neurons)))
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])

        # Step 4: Step through the environment
        next_observation, reward, terminated, truncated, _ = env.step([action])
        done = terminated or truncated
        total_reward += reward

        # Step 5: Update weights using the EH rule
        weights = update_weights(weights, input_activities, raw_activities, activation_mean, reward, reward_mean)

        # Step 6: Update running means (low-pass filtered)
        activation_mean = 0.8 * activation_mean + 0.2 * raw_activities
        reward_mean = 0.8 * reward_mean + 0.2 * reward

        # Move to the next observation
        observation = next_observation
        step += 1

        # Anneal exploration noise
        exploration_level = max(0.01, exploration_level * 0.999)

    # Store reward for analysis
    reward_history.append(total_reward)

    # Print progress
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Exploration Level: {exploration_level:.4f}")

# Plot results
import matplotlib.pyplot as plt

# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(reward_history, label="Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("reward_plotv2.png")