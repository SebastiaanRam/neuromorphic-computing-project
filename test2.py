import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def get_s_neuron_activity(a_input):
    """Compute neuron activities based on input signals with a threshold-linear function."""
    return np.maximum(0, a_input)

def get_a_synaptic_input(w, x, noise_variance):
    """Compute synaptic input with noise added to the weights."""
    noise = np.random.normal(0, np.sqrt(noise_variance), size=w.shape)
    return np.dot(w + noise, x)

def get_x_neuron_activity(y_star, weights, d, n_total):
    """Compute input neuron activities mapping to motor cortex targets."""
    y_star = y_star / np.linalg.norm(y_star)  # Normalize desired movement direction

    Q = np.random.uniform(-1, 1, (d, n_total))
    Q_inverse = np.linalg.pinv(Q)  # Inverse mapping matrix
    s_tilde = np.dot(Q_inverse, y_star)  # Target motor cortex activities

    c_rate = 10  # Scaling for firing rates
    x_activities = c_rate * np.dot(np.linalg.pinv(weights), s_tilde)  # Input activities
    return np.maximum(0, x_activities)  # Apply threshold-linear activation

def get_cursor_velocity(s_activities, p_directions, k_s=0.03, alpha=1, beta=0, d=2):
    """Compute cursor velocity from neuron activities and preferred directions."""
    normalized_activities = (s_activities - beta) / alpha
    population_vector = np.sum(
        (normalized_activities[:, np.newaxis] * p_directions), axis=0
    )
    y_t = k_s * (d / len(s_activities)) * population_vector
    return y_t

def r_ang(y, y_star):
    """Compute alignment (reward) between movement and desired direction."""
    if np.linalg.norm(y) == 0 or np.linalg.norm(y_star) == 0:
        return 0  # Avoid division by zero
    return np.dot(y, y_star) / (np.linalg.norm(y) * np.linalg.norm(y_star))

def low_pass_filter(previous, current, alpha=0.2):
    """Low-pass filter to smooth signals."""
    return (1 - alpha) * previous + alpha * current

def update_weights(weights, x_activities, a_input, a_input_hat, R_t, R_hat, learning_rate):
    """Update synaptic weights using the EH learning rule."""
    a_difference = a_input - a_input_hat
    r_difference = R_t - R_hat
    delta_weights = learning_rate * np.outer(x_activities, a_difference).T * r_difference
    return weights + delta_weights

# Environment setup
env = gym.make("LunarLander-v2", render_mode="human")

# Simulation parameters
max_t = 100
biological_time = 1 / 30  # Timestep in biological time
learning_rate = 0.01
variance = 1  # Noise variance
n_input = 8  # Input neurons
n_total = 4  # Total motor neurons
n_recorded = 4  # Recorded neurons
d = 2  # Dimensionality of movement space
weights = np.random.uniform(-0.5, 0.5, (n_total, n_input))

# Target directions and positions
target_position_l = np.array([0, 0])
target_directions = np.array([[1, 1], [1, 0], [-1, 1], [-1, 0]])
target_directions = target_directions / np.linalg.norm(target_directions, axis=1, keepdims=True)

R_t = 0
old_a_input = np.zeros(n_total)
total_rewards = []

# Run simulation
episodes = 1000
for i in range(episodes):
    state, info = env.reset()
    action = 1
    episode_reward = 0

    for t in np.arange(0, max_t * biological_time, biological_time):
        current_position_l = np.array([state[0], state[1]])
        next_state, reward, done, truncated, info = env.step(action)
        if done:
            total_rewards.append(episode_reward)
            print(f"Episode {i+1} finished with reward: {episode_reward}")
            break
        episode_reward += reward

        # Compute desired direction
        y_star = target_position_l - current_position_l
        if np.linalg.norm(y_star) == 0:
            y_star = np.ones_like(y_star)  # Avoid zero vector
        y_star = y_star / np.linalg.norm(y_star)  # Normalize

        # Compute input activities
        x_activities = get_x_neuron_activity(y_star, weights, d, n_total)

        # Compute synaptic and motor neuron activities
        a_input = get_a_synaptic_input(weights, x_activities, variance)
        s_activities = get_s_neuron_activity(a_input)

        # Compute cursor velocity
        y_t = get_cursor_velocity(s_activities, target_directions)

        # Compute reward signal
        new_R_t = r_ang(y_t, y_star)
        R_hat = low_pass_filter(R_t, new_R_t)
        R_t = new_R_t

        # Smooth synaptic input
        a_input_hat = low_pass_filter(old_a_input, a_input)
        old_a_input = a_input

        # Update weights
        weights = update_weights(weights, x_activities, a_input, a_input_hat, R_t, R_hat, learning_rate)

        # Determine next action
        action = np.argmax(s_activities)
        state = next_state

env.close()

# Plot rewards
plt.figure(figsize=(10, 6))
plt.plot(total_rewards, label='Episode Rewards', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards Over Episodes')
plt.legend()
plt.grid(True)
plt.show()
