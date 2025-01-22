import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers

# Set up the Lunar Lander environment
env = gym.make("LunarLander-v2")
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

# Define the neural network
class LunarLanderNetwork(tf.keras.Model):
    def __init__(self, num_inputs, num_actions):
        super(LunarLanderNetwork, self).__init__()
        self.hidden1 = layers.Dense(32, activation="relu")
        # self.hidden2 = layers.Dense(64, activation="relu")
        self.output_layer = layers.Dense(num_actions, activation="linear")

    def call(self, inputs):
        x = self.hidden1(inputs)
        # x = self.hidden2(x)
        return self.output_layer(x)

class RewardModulatedHebbian:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.reward_mean = 0
        self.decay_factor = 0.99

    def update_weights(self, inputs, activations, reward):
        # Compute the reward signal deviation from the mean
        reward_delta = reward - self.reward_mean
        self.reward_mean = self.decay_factor * self.reward_mean + (1 - self.decay_factor) * reward

        # Forward pass through the network to get intermediate activations
        x = inputs
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights, biases = layer.get_weights()

                # Compute pre- and post-synaptic activity
                pre_synaptic = x.numpy() if tf.is_tensor(x) else x  # Ensure pre_synaptic is a NumPy array
                post_synaptic = layer(x).numpy() if tf.is_tensor(layer(x)) else layer(x)  # Ensure post_synaptic is a NumPy array

                # Compute weight update
                weight_update = (
                    self.learning_rate * reward_delta * np.matmul(pre_synaptic.T, post_synaptic - np.mean(post_synaptic, axis=0))
                )

                # Update weights
                weights += weight_update
                layer.set_weights([weights, biases])

                # Pass the output to the next layer
                x = post_synaptic



# Initialize the network and Hebbian learning mechanism
model = LunarLanderNetwork(num_inputs, num_actions)
hebbian_learning = RewardModulatedHebbian(model, learning_rate=0.01)

# Define the training loop
def train_lunar_lander(env, model, hebbian_learning, episodes=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, num_inputs])
        total_reward = 0

        while True:
            # Get action probabilities
            logits = model(state)
            action_probs = tf.nn.softmax(logits)
            action = np.random.choice(num_actions, p=action_probs.numpy()[0])

            # Perform action
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, num_inputs])
            total_reward += reward

            # Update weights with Hebbian learning
            hebbian_learning.update_weights(state, logits.numpy(), reward)

            state = next_state
            
            if done:
                print(f"Episode {episode + 1}: Total Reward: {total_reward}")
                break

# Train the model
train_lunar_lander(env, model, hebbian_learning, episodes=5000)