import gymnasium as gym
import matplotlib
import numpy as np
matplotlib.use('Agg')  # Use a backend suitable for your environment
import matplotlib.pyplot as plt

def low_pass_filter(previous, current, alpha=0.8):
    """Apply a low-pass filter to smooth values."""
    return (1 - alpha) * previous + alpha * current

def activation_function(x):
    """Threshold-linear activation function."""
    return np.maximum(0, x)

def reward_signal(state, reward):
    """Calculate a reward-modulated signal based on environment feedback."""
    angle_penalty = abs(state[4]) * 10  # Penalize tilting too far
    speed_penalty = np.linalg.norm([state[2], state[3]]) * 5  # Penalize excessive speed
    return reward - angle_penalty - speed_penalty

# Environment setup
env = gym.make("LunarLander-v2")
n_input = env.observation_space.shape[0]  # Number of state features
n_output = env.action_space.n  # Number of actions
weights = np.random.uniform(-0.5, 0.5, (n_input, n_output))  # Initial weights
learning_rate = 1e-3
exploration_level = 0.1

# Training parameters
episodes = 10000
max_steps = 300
all_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0
    R_t, a_input_prev = 0, 0
    done = False

    for _ in range(max_steps):
        if done:
            break

        # Neural network activation
        a_input = np.dot(state, weights) + np.random.normal(0, exploration_level, n_output)
        s_activity = activation_function(a_input)
        action = np.argmax(s_activity)

        # Environment step
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        next_state = np.array(next_state)

        # Desired movement direction
        y_star = -state[0:2]  # Target position to (0, 0) for landing
        y_star /= np.linalg.norm(y_star) if np.linalg.norm(y_star) > 0 else 1
        cursor_velocity = np.array([state[2], state[3]])  # Approximate cursor velocity

        # Calculate reward-modulated signal
        reward_mod = reward_signal(state, reward)
        R_hat = low_pass_filter(R_t, reward_mod)
        R_t = reward_mod

        a_input_hat = low_pass_filter(a_input_prev, a_input)
        a_input_prev = a_input

        # Weight update using EH-rule
        delta_w = learning_rate * np.outer(state, (a_input - a_input_hat)) * (R_t - R_hat)
        weights += delta_w

        # Move to the next state
        state = next_state

    all_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{episodes}: Total Reward = {total_reward}")

# Plotting rewards
plt.figure(figsize=(10, 6))
plt.plot(all_rewards, label="Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress with Reward-Modulated Hebbian Learning")
plt.legend()
plt.grid()
plt.savefig("reward_plot.png")

env.close()
