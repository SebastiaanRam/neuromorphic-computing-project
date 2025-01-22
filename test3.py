import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("LunarLander-v2")
input_neurons = env.observation_space.shape[0]  # m = 8
motor_neurons = env.action_space.n             # n_total = 4

# Hyperparameters
learning_rate = 0.01
exploration_level = 0.5
num_episodes = 5000
max_steps = 1000

# Initialize weights
weights_input_to_x = np.random.uniform(-0.5, 0.5, (input_neurons, input_neurons))
weights_x_to_motor = np.random.uniform(-0.5, 0.5, (motor_neurons, input_neurons))

def get_synaptic_input(weights, inputs, noise_std):
    noise = np.random.normal(0, noise_std, weights.shape[0])
    return np.dot(weights, inputs) + noise

def activation_function(inputs):
    return np.maximum(0, inputs)  # ReLU activation

def low_pass_filter(prev, curr, alpha=0.8):
    return (1 - alpha) * prev + alpha * curr

# Training
rewards = []
for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    previous_a_input = np.zeros(motor_neurons)
    previous_reward = 0

    for step in range(max_steps):
        # Compute input population activities (x_activities)
        x_activities = activation_function(get_synaptic_input(weights_input_to_x, obs, exploration_level))

        # Compute motor cortex activities
        a_input = get_synaptic_input(weights_x_to_motor, x_activities, exploration_level)
        s_activities = activation_function(a_input)

        # Choose action based on max activity
        action = np.argmax(s_activities)

        # Step in the environment
        next_obs, reward, done, _, _ = env.step(action)
        episode_reward += reward

        # Reward-modulated Hebbian update for motor layer
        filtered_a_input = low_pass_filter(previous_a_input, a_input)
        filtered_reward = low_pass_filter(previous_reward, reward)

        delta_weights_x_to_motor = learning_rate * np.outer(
            (a_input - filtered_a_input),  # Postsynaptic deviation
            x_activities                 # Presynaptic activity
        ) * (reward - filtered_reward)  # Reward deviation
        weights_x_to_motor += delta_weights_x_to_motor

        # Reward-modulated Hebbian update for input-to-x layer (optional)
        delta_weights_input_to_x = learning_rate * np.outer(
            (x_activities - low_pass_filter(np.zeros_like(x_activities), x_activities)),  # Postsynaptic deviation
            obs  # Presynaptic activity
        ) * (reward - filtered_reward)
        weights_input_to_x += delta_weights_input_to_x

        obs = next_obs
        previous_a_input = a_input
        previous_reward = reward

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
plt.savefig("rewards_plot_with_x_activities.png")

cumsum_rewards = np.cumsum(rewards)
plt.figure(figsize=(10, 6))
plt.plot(cumsum_rewards, label="Cumulative Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("cumsum_rewards_with_x_activities.png")
env.close()
