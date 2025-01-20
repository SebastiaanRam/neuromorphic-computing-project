import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("LunarLander-v2")
input_neurons = env.observation_space.shape[0]  # m = 8
hidden_neurons = 16                             # Hidden layer size
motor_neurons = env.action_space.n             # n_total = 4

# Hyperparameters
learning_rate = 0.005
exploration_level_max = 1.0
exploration_level_min = 0.1
decay_rate = 0.001  # Decay rate for exploration decay
use_exploration_decay = True
exploration_level = exploration_level_max
num_episodes = 5000
max_steps = 1000

# Initialize weights
weights_input_hidden = np.random.uniform(-0.1, 0.1, (hidden_neurons, input_neurons))
weights_hidden_output = np.random.uniform(-0.1, 0.1, (motor_neurons, hidden_neurons))

def get_synaptic_input(weights, inputs, variance):
    """
    Compute synaptic input to neurons with scaled noise for stability.
    """
    noise = np.random.normal(0, variance, weights.shape[0])
    return np.dot(weights, inputs) + noise 
    
def activation_function(inputs):
    """
    Activation function: ReLU with clipping to prevent excessively large activities.
    """
    return np.clip(np.maximum(0, inputs), 0, 1e3)  # Clip max activity to 1000

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
    norms = np.maximum(norms, 1e-8)  
    norms = np.clip(norms, a_min=None, a_max=max_norm)
    return weights / norms


# Training
rewards = []

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    previous_hidden_input = np.zeros(hidden_neurons)
    previous_output_input = np.zeros(motor_neurons)
    previous_reward = 0

    for step in range(max_steps):
        # Forward pass: Input to Hidden Layer
        hidden_input = get_synaptic_input(weights_input_hidden, obs, exploration_level)
        hidden_activities = activation_function(hidden_input)

        # Forward pass: Hidden to Output Layer
        output_input = get_synaptic_input(weights_hidden_output, hidden_activities, exploration_level)
        output_activities = activation_function(output_input)

        # Choose action based on max activity
        action = np.argmax(output_activities)

        # Step in the environment
        next_obs, reward, done, _, _ = env.step(action)
        episode_reward += reward

        # Reward-modulated Hebbian update for Hidden-to-Output weights
        filtered_output_input = low_pass_filter(previous_output_input, output_input)
        filtered_reward = low_pass_filter(previous_reward, reward)

        delta_weights_hidden_output = learning_rate * np.outer(
                (output_input - filtered_output_input),  # Postsynaptic deviation
                hidden_activities                       # Presynaptic activity
            ) * (reward - filtered_reward)             # Reward deviation
        
        weights_hidden_output += delta_weights_hidden_output
        weights_hidden_output = normalize_weights(weights_hidden_output)

        # Reward-modulated Hebbian update for Input-to-Hidden weights
        filtered_hidden_input = low_pass_filter(previous_hidden_input, hidden_input)

        delta_weights_input_hidden = learning_rate * np.outer(
                (hidden_input - filtered_hidden_input),  # Postsynaptic deviation
                obs                                     # Presynaptic activity
            ) * (reward - filtered_reward)             # Reward deviation
        weights_input_hidden += delta_weights_input_hidden
        weights_input_hidden = normalize_weights(weights_input_hidden)

        # Update state variables for the next step
        obs = next_obs
        previous_hidden_input = hidden_input
        previous_output_input = output_input
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
plt.title("X activities with weight update")
plt.grid()
plt.savefig("kindaworking/x_activities_with_weight_update_cumsum.png")


# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(rewards, label="Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("X activities with weight update")
plt.legend()
plt.grid()
plt.savefig("kindaworking/x_activities_with_weight_update_rewards.png")


env.close()
