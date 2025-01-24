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
# Network consists of two populations of neurons connected in a feedforward manner

# First population: Input population to neurons in the motor cortex
m = 8
# Activities x1(t), . . . ,xm(t)

# Second population: Neurons in the motor cortex that receive inputs from the input population
n_total = 4
# Activities s1(t), . . . ,sntotal

# Learning takes place solely in the synapses of synapses of projections between these populations

# s. In principle, the same learning is applicable to multilayer networks!!!
# This means we could implement the code in a multilayer network 

# In the paper all the n neurons were used to determine the monkey arm movement in our model
# However, only n 40 of these (the “recorded” subset) were used for cursor control
# This probably isn't the case in our model

# Randomize weights at the beginning of the each simulation, drawn from a uniform distribution between -0.5 and 0.5
# Not sure if it should be m, n_total or n_total, m
hidden_neurons = 4
x_weights = np.random.uniform(-0.5, 0.5, (hidden_neurons, m))
a_weights = np.random.uniform(-0.5, 0.5, (n_total, hidden_neurons))


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

def compute_x_activities(state, W_input, noise_level=0.01):
    # Linear mapping from state to activities
    x = np.dot(W_input, state)
    
    # Optional: Apply ReLU activation
    # x = np.maximum(x, 0)
    
    # Add exploratory noise
    # noise = np.random.uniform(-noise_level, noise_level, size=x.shape)
    # x += noise
    
    return x

def low_pass_filter(previous, current, alpha=0.8):
    """Low pass filter to smooth values"""
    return (1 - alpha) * previous + alpha * current

previous_a_input = np.zeros(n_total)
previous_reward = 0

max_episode_steps = 1000
num_episodes = 1000
learning_rate = 0.01
episode_rewards = []
for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    print(f"Observation: {observation}")
    for step in range(max_episode_steps):
        # Compute activities of input population
        x_acitivities = compute_x_activities(observation, x_weights)

        # Compute activities of motor cortex population
        current_a_input = get_a_synaptic_input(a_weights, x_acitivities)

        motor_activities = get_s_neuron_activity(current_a_input)

        action = np.argmax(motor_activities)
        print(f"Action: {action}")
        # Take a step in the environment
        next_observation, current_reward, done, _, _ = env.step(action)
        episode_reward += current_reward

        # Compute activities of motor cortex population
        filtered_a_input = low_pass_filter(previous_a_input, current_a_input, alpha=0.8) 
        filtered_reward = low_pass_filter(previous_reward, current_reward, alpha=0.8) 

        # Compute the difference between current and filtered values
        a_difference = current_a_input - filtered_a_input
        r_difference = current_reward - filtered_reward

        print(f"a_difference: {a_difference.shape}")
        print(f"x_acitivities: {x_acitivities.shape}")
        print(f"r_difference: {r_difference}")

        # Compute the delta weights
        delta_weights = learning_rate * x_acitivities * a_difference * r_difference

        # Update weights
        a_weights += delta_weights

        # Update observation for the next step
        observation = next_observation
        previous_a_input = current_a_input
        previous_reward = current_reward
        if done:
            break

    episode_rewards.append(episode_reward)

# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("reward_plot_test.png")

env.close()