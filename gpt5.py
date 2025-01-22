import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')  # Use a backend suitable for your environment
import matplotlib.pyplot as plt

class LunarLanderNetwork:
    def __init__(self, input_dim, hidden_dim, recorded_dim, learning_rate, noise_level):
        """
        Initialize the feedforward network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.recorded_dim = recorded_dim
        self.learning_rate = learning_rate
        self.noise_level = noise_level

        # Initialize weights randomly
        self.weights = np.random.uniform(-0.5, 0.5, (hidden_dim, input_dim))

        # Placeholder for filtered activations and rewards
        self.filtered_activations = np.zeros(hidden_dim)
        self.filtered_reward = 0.0

    def relu(self, x):
        """
        Threshold linear activation (ReLU).
        """
        return np.maximum(0, x)

    def softmax(self, x):
        """
        Softmax activation function for action probabilities.
        """
        exp_x = np.exp(x - np.max(x))  # Stability improvement
        return exp_x / exp_x.sum()

    def forward(self, inputs):
        """
        Perform a forward pass through the network.
        """
        # Compute hidden layer activations
        total_synaptic_input = self.weights @ inputs + np.random.normal(0, self.noise_level, self.hidden_dim)
        activations = self.relu(total_synaptic_input)
        return activations

    def update_weights(self, inputs, activations, reward):
        """
        Update weights using the exploratory Hebbian learning rule.
        """
        # Update filtered activations and reward
        self.filtered_activations = 0.8 * self.filtered_activations + 0.2 * activations
        self.filtered_reward = 0.8 * self.filtered_reward + 0.2 * reward

        # Hebbian weight update rule
        for i in range(self.hidden_dim):
            for j in range(self.input_dim):
                self.weights[i, j] += self.learning_rate * inputs[j] * \
                                      (activations[i] - self.filtered_activations[i]) * \
                                      (reward - self.filtered_reward)

    def choose_action(self, activations):
        """
        Compute action probabilities and select an action.
        """
        # Use the first 'recorded_dim' neurons for action probabilities
        recorded_activations = activations[:self.recorded_dim]
        action_probabilities = self.softmax(recorded_activations)
        return np.random.choice(len(action_probabilities), p=action_probabilities)


# Initialize the Lunar Lander environment and network
env = gym.make("LunarLander-v2", render_mode="human")
max_steps = 1000
network = LunarLanderNetwork(input_dim=8, hidden_dim=4, recorded_dim=40, learning_rate=1e-3, noise_level=0.1)
total_rewards = []
# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    obs, env_state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    while not done and steps < max_steps:
        # Forward pass to get activations
        activations = network.forward(obs)

        # Choose an action
        action = network.choose_action(activations)

        # Perform the action in the environment
        next_obs, env_state, reward, done, _ = env.step(action)
        total_reward += reward
        # print(f"Done {done}")
        # Update weights based on reward
        network.update_weights(obs, activations, reward)

        # Move to the next state
        obs = next_obs
        steps += 1
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    total_rewards.append(total_reward)

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_episodes + 1), total_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Rewards Over Episodes")
plt.legend()
plt.grid()
plt.savefig("reward_plot.png")
env.close()