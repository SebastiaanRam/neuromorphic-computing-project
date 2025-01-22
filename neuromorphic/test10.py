import gymnasium as gym
import numpy as np

# Hyperparameters
LEARNING_RATE = 0.01
EXPLORATION_NOISE = 0.1
DISCOUNT_FACTOR = 0.99
EPISODES = 1000

# RMHL implementation class
class RMHLAgent:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.uniform(-0.5, 0.5, (input_dim, output_dim))
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.filtered_reward = 0
        self.filtered_activation = np.zeros(output_dim)
        self.reward_smoothing = 0.8
        self.activation_smoothing = 0.8

    def act(self, state):
        activation = np.dot(state, self.weights)
        noise = np.random.uniform(-EXPLORATION_NOISE, EXPLORATION_NOISE, self.output_dim)
        noisy_activation = activation + noise
        return np.tanh(noisy_activation), noise

    def update_weights(self, state, activation, noise, reward):
        self.filtered_reward = (self.reward_smoothing * self.filtered_reward + 
                               (1 - self.reward_smoothing) * reward)

        self.filtered_activation = (self.activation_smoothing * self.filtered_activation + 
                                    (1 - self.activation_smoothing) * activation)

        reward_deviation = reward - self.filtered_reward
        activation_deviation = activation - self.filtered_activation

        gradient = np.outer(state, activation_deviation * reward_deviation)
        self.weights += LEARNING_RATE * gradient

# Initialize environment and agent
env = gym.make('Pendulum-v1')
input_dim = env.observation_space.shape[0]
output_dim = 1  # Torque action is a single continuous value
agent = RMHLAgent(input_dim, output_dim)

# Training loop
for episode in range(EPISODES):
    state = env.reset()
    state = state[0]  # To support Gymnasium's reset API
    total_reward = 0
    done = False
    max_steps = 200
    step = 0
    while not done and step < max_steps:
        action, noise = agent.act(state)
        next_state, reward, done, _, _ = env.step(action * 2)  # Scale action to [-2, 2]

        total_reward += reward
        agent.update_weights(state, action, noise, reward)

        step += 1
        state = next_state

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()
