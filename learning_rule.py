import gymnasium as gym
import numpy as np

def get_s_neuron_activity(a_input):
    """Compute neuron activities with threshold-linear activation."""
    return np.maximum(0, a_input)

def get_a_synaptic_input(w, x, noise_std):
    """Compute synaptic input to motor cortex neurons with scaled noise."""
    noise = np.random.normal(0, noise_std, size=w.shape[0])
    return np.dot(w, x) + noise

def get_x_neuron_activity(y_star, weights):
    """Determine input neuron activities."""
    y_star = y_star / np.linalg.norm(y_star)  # Normalize target direction
    Q = np.random.uniform(-1, 1, (d, n_total))  # Connectivity matrix
    Q_inverse = np.linalg.pinv(Q)
    s_tilde = np.dot(Q_inverse, y_star)
    c_rate = 10
    x_activities = c_rate * np.dot(np.linalg.pinv(weights), s_tilde)
    return np.maximum(0, x_activities)

def get_cursor_velocity(s_activities, k_s=0.03):
    """
    Compute cursor velocity from neuron activities.
    Scaled to influence the choice of actions.
    """
    return k_s * s_activities / np.sum(s_activities + 1e-6)  # Normalize and scale

def r_ang(y, y_star):
    """Calculate alignment between movement and desired direction."""
    y = y / np.linalg.norm(y)
    y_star = y_star / np.linalg.norm(y_star)
    return np.dot(y, y_star)

def low_pass_filter(previous, current, alpha=0.2):
    """Apply low-pass filtering."""
    return (1 - alpha) * previous + alpha * current

# Environment setup
env = gym.make("LunarLander-v3", render_mode="human")
n_input = 8  # Observation space dimensions
n_total = 4  # Number of actions
learning_rate = 0.01
variance = 1.0  # Noise variance
episodes = 500

# Initialize weights and actions
weights = np.random.uniform(-0.5, 0.5, (n_total, n_input))
actions = [0, 1, 2, 3]  # Discrete actions

# Simulation loop
for episode in range(episodes):
    state, info = env.reset()
    total_rewards = []
    old_a_input = np.zeros(n_total)
    R_t = 0

    print(f"Starting Episode {episode + 1}")

    for t in range(1000):  # Limit steps per episode
        y_star = np.zeros(n_input)  # Target is zero velocity, upright position
        y_star[0:2] = [0, 0]  # Target position (center, lander pad)
        y_star[2:4] = [0, 0]  # Target velocity (zero)

        x_activities = state  # Input neuron activities are the state
        # print(f"Input neuron activities (x_activities): {x_activities}")

        a_input = get_a_synaptic_input(weights, x_activities, variance)
        # print(f"Synaptic input (a_input): {a_input}")

        s_activities = get_s_neuron_activity(a_input)

        # Calculate cursor velocity (influence for actions)
        cursor_velocity = get_cursor_velocity(s_activities)
        print(f"Cursor velocity (y_t): {cursor_velocity}")

        # Select action based on neuron activities and velocity
        weighted_actions = cursor_velocity * s_activities
        action = np.argmax(weighted_actions)

        # Take a step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        total_rewards.append(reward)

        # EH rule updates
        new_R_t = reward
        R_hat = low_pass_filter(R_t, new_R_t)
        R_t = new_R_t

        a_input_hat = low_pass_filter(old_a_input, a_input)
        old_a_input = a_input

        a_difference = a_input - a_input_hat
        r_difference = R_t - R_hat

        delta_weights = learning_rate * np.outer(a_difference, x_activities)
        weights += delta_weights
        # print(f"Weight update (delta_weights): {delta_weights}")
        # print(f"Updated weights: {weights}")

        if done or truncated:
            print(f"Episode {episode + 1} ended after {t + 1} steps.")
            print(f"Total reward: {np.sum(total_rewards)}")
            break

        state = next_state

    variance *= 0.99  # Decay noise over episodes

env.close()