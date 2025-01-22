import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Environment setup
env = gym.make("CartPole-v1")
actions = env.action_space.n
observations = env.observation_space.shape[0]

# Hyperparameters
hidden_neurons = 32
learning_rate = 0.01
noise_variance = 0.1
exploration_decay = 0.999
max_episode_steps = 500
num_episodes = 5000

# Weight initialization
x_weights = np.random.uniform(-0.5, 0.5, (hidden_neurons, observations))
a_weights = np.random.uniform(-0.5, 0.5, (actions, hidden_neurons))

def low_pass_filter(previous, current, alpha=0.8):
    """Smooth values using an exponential moving average."""
    return (1 - alpha) * previous + alpha * current

def compute_x_activities(state, W_input):
    """Compute activities for the x-layer with ReLU activation."""
    x = np.dot(W_input, state)
    return np.maximum(0, x)

def get_a_synaptic_input(w, x, noise_std):
    """Compute synaptic input to motor cortex neurons."""
    noise = np.random.normal(0, noise_std, w.shape[0])
    return np.dot(w, x) + noise

def normalize_weights(weights):
    """Normalize weights to maintain stability."""
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    return weights / np.maximum(norms, 1e-8)

episode_rewards = []

for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0

    # Initialize filtered values
    previous_a_input = np.zeros(actions)
    previous_reward = 0

    for step in range(max_episode_steps):
        # Compute x-layer activities
        x_activities = compute_x_activities(observation, x_weights)

        # Compute motor cortex activities
        current_a_input = get_a_synaptic_input(a_weights, x_activities, np.sqrt(noise_variance))
        motor_activities = np.maximum(0, current_a_input)  # ReLU activation

        # Select action based on motor cortex output
        action = np.argmax(motor_activities)

        # Step in the environment
        next_observation, current_reward, done, _, _ = env.step(action)
        episode_reward += current_reward

        # Reward shaping
        pole_angle = abs(next_observation[2])
        pole_velocity = abs(next_observation[3])
        shaped_reward = current_reward - 0.1 * (pole_angle + pole_velocity)

        # Temporal filtering
        filtered_a_input = low_pass_filter(previous_a_input, current_a_input)
        filtered_reward = low_pass_filter(previous_reward, shaped_reward)

        # Compute differences
        a_difference = current_a_input - filtered_a_input
        r_difference = shaped_reward - filtered_reward

        # Update weights (reward-modulated Hebbian learning)
        delta_weights = learning_rate * np.outer(a_difference, x_activities) * r_difference
        a_weights += delta_weights

        # Normalize weights
        a_weights = normalize_weights(a_weights)

        # Update states
        observation = next_observation
        previous_a_input = current_a_input
        previous_reward = shaped_reward

        if done:
            break

    # Decay noise for exploration-exploitation tradeoff
    noise_variance *= exploration_decay
    episode_rewards.append(episode_reward)

# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Episode Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress with Enhanced Hebbian Learning (CartPole-v1)")
plt.legend()
plt.grid()
plt.savefig("enhanced_cartpole_rewards_plot.png")
env.close()
