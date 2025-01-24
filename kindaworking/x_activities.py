from typing import no_type_check_decorator
import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use a backend suitable for your environment
import matplotlib.pyplot as plt

# Environment setup
env = gym.make("LunarLander-v3")
actions = env.action_space.n
observations = env.observation_space.shape[0]
print(f"Actions: {actions}")
print(f"Observations: {observations}")
m = observations
n_total = actions
hidden_neurons = 16
x_weights = np.random.uniform(-0.1, 0.1, (hidden_neurons, m))
a_weights = np.random.uniform(-0.1, 0.1, (n_total, hidden_neurons))


def get_a_synaptic_input(w, x, v=1):
    """
    Compute synaptic input to motor cortext neurons. With noise added to the weights
    w is the synaptic efficacy from input neuron j to i
    Noise used for exploration, drawn from a zero mean distribution with v being variance(exploration_level).
    """

    return np.dot(w, x) + np.random.normal(0, v)  

def get_s_neuron_activity(a_input):
    """
    Activity of s of neuron i at time t was modeled as a non-linear function of the synaptic input a_input
    The activation function ensures there are no negative activities
    """
    return np.maximum(0, a_input)

def compute_x_activities(state, W_input):
    # Linear mapping from state to activities
    x = np.dot(W_input, state)
    return x

def low_pass_filter(previous, current, alpha=0.8):
    """Low pass filter to smooth values"""
    return (1 - alpha) * previous + alpha * current

def normalize_weights(weights, max_norm=1.0):
    """
    Normalize weights to prevent unbounded growth.
    """
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    norms = np.clip(norms, a_min=None, a_max=max_norm)
    return weights / norms

previous_a_input = np.zeros(n_total)
previous_reward = 0

max_episode_steps = 1000
num_episodes = 5000
learning_rate = 0.005
episode_rewards = []
for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    # print(f"Observation: {observation}")
    for step in range(max_episode_steps):
        # Compute activities of input population
        x_activities = compute_x_activities(observation, x_weights)

        # Compute activities of motor cortex population
        current_a_input = get_a_synaptic_input(a_weights, x_activities)

        motor_activities = get_s_neuron_activity(current_a_input)

        action = np.argmax(motor_activities)

        # Take a step in the environment
        next_observation, current_reward, done, _, _ = env.step(action)
        episode_reward += current_reward

        # Compute activities of motor cortex population
        filtered_a_input = low_pass_filter(previous_a_input, current_a_input, alpha=0.8) 
        filtered_reward = low_pass_filter(previous_reward, current_reward, alpha=0.8) 

        # Compute the difference between current and filtered values
        a_difference = current_a_input - filtered_a_input
        r_difference = current_reward - filtered_reward
        # Compute the delta weights
        delta_weights = learning_rate * np.outer(a_difference, x_activities) * r_difference

        # Update weights
        a_weights += delta_weights
        a_weights = normalize_weights(a_weights)

        # Update observation for the next step
        observation = next_observation
        previous_a_input = current_a_input
        previous_reward = current_reward

        if done:
            break

    episode_rewards.append(episode_reward)

cumsum_rewards = np.cumsum(episode_rewards)
plt.plot(cumsum_rewards)
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.title("X activities without weight update")
plt.grid()
plt.savefig("kindaworking/x_without_weight_update_cumsum.png")

# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("kindaworking/x_without_weight_update_rewards.png")

env.close()