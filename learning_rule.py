import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def get_s_neuron_activity(a_input):
    """Compute neuron activities based on input signals."""
    return np.maximum(0, a_input)


def get_a_synaptic_input(w, x):
    """Compute synaptic input to motor cortext neurons. With noise added to the weights"""
    return np.dot(w, x) #+ get_noise()


def get_x_neuron_activity(y_star, weights, n_input, d):
    """
    Calculate the input neuron activities based on the desired movement direction.

    Parameters:
    - y_star: Desired movement direction (shape: [d]) - normalized.
    - weights: Weight matrix connecting input neurons to motor cortex neurons (shape: [n_input, n_total]).
    - n_input: Number of input neurons.
    - d: Dimensionality of the movement space (typically 2 for 2D movement).

    Returns:
    - x_activities: Activities of the input neurons (shape: [n_input]).
    """

    # Step 1: Normalize the desired movement direction y_star (if not already normalized)
    y_star = y_star / np.linalg.norm(y_star)

    # Step 2: Generate a random matrix Q representing the mapping from y_star to input neuron activities
    Q = np.random.uniform(-1, 1, (d, n_input))  # Mapping matrix from y_star to input neuron activities
    Q_inverse = np.linalg.pinv(Q)  # Inverse of the mapping matrix

    # Step 3: Compute the intermediate target activities (s_tilde) in the motor cortex
    s_tilde = np.dot(Q_inverse, y_star)  # Target activities based on the desired direction

    # Step 4: Compute the input neuron activities (x_activities)
    # Use the inverse of the weights to map from s_tilde back to input neuron activities
    x_activities = np.dot(np.linalg.pinv(weights), s_tilde)  # Inverse mapping of motor cortex activities

    return x_activities


def delta(x):
    """Threshold-linear activation function that ensures non-negative activities."""
    return max(0, x)


def get_noise():
    """Generate noise from a zero-mean distribution."""
    return np.random.normal(0, np.sqrt(variance))


def get_cursor_velocity(s_activities, p_directions, k_s=0.03, alpha=1, beta=0, d=2):
    """
    Compute the cursor velocity from neuron activities.

    Parameters:
    - s_activities: Array of neuron activities (shape: [n_neurons]).
    - p_directions: Array of preferred directions for neurons (shape: [n_neurons, d]).
    - k_s: Speed factor (default: 0.03).
    - alpha: Normalization scaling factor for activities.
    - beta: Offset factor for activities.
    - d: Dimensionality of the movement space (e.g., 2 for 2D).

    Returns:
    - y_t: Cursor velocity vector (shape: [d]).
    """
    # Normalize activities
    normalized_activities = (s_activities - beta) / alpha

    # Ensure p_directions matches the shape of s_activities
    if len(normalized_activities) != p_directions.shape[0]:
        raise ValueError(f"The number of neurons in 's_activities' and 'p_directions' must match. Normalized activities: {len(normalized_activities)} and p-directions: {p_directions.shape[0]}")

    # Compute the population vector (direction weighted by activity)
    population_vector = np.sum(
        normalized_activities[:, np.newaxis] * p_directions, axis=0
    )

    # Scale by the speed factor and dimensionality
    y_t = k_s * (d / len(s_activities)) * population_vector
    return y_t * 100 #was getting very small cursor velocity
   



def r_ang(y, y_star, reward):
    """Angle between movement direction and desired direction"""
    # r_t = reward + np.dot(y, y_star) * np.linalg.norm(y) * 0.01  -- quite good
    r_t = reward + np.dot(y, y_star) * np.linalg.norm(y)
    return r_t


def low_pass_filter(previous, current, alpha=0.8):
    """Low pass filter to smooth values"""
    return (1 - alpha) * previous + alpha * current

def select_action(state, weights):
    """
    Select an action based on state and weights using dot product.
    - state: Current state (input features)
    - weights: Current weights matrix
    """
    action_values = np.dot(state, weights)  # Calculate output activations
    action = np.argmax(action_values)
    return action



# env = gym.make("LunarLander-v3")
env = gym.make("LunarLander-v3",  render_mode="human")

# Simulation params
max_t = 3000
# ,a timestep in our simulation corresponded to 1/30s in biological time.
biological_time = 1 / 30
time_steps = np.arange(0, max_t * biological_time, biological_time)
learning_rate = 0.1
learning_rate_min = 0.001  # Minimum exploration rate
learning_rate_decay = 0.995  # Decay factor
variance = 1  # Variance for the noise distribution
# print(f"Observation space {env.observation_space}")
n_input = 8  # Observation space.
n_total = 4
d = 2  # movement dimensionality
weights = np.random.rand(n_input, n_total) * 0.001

target_position_l = np.array([0, 0])
# 4 corners of the gridworld
target_directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

# Calculate intermediate directions
intermediate_directions = np.array([
    (target_directions[0] + target_directions[1]) / 2,  # Between [1, 1] and [1, -1]
    (target_directions[1] + target_directions[3]) / 2,  # Between [1, -1] and [-1, -1]
    (target_directions[3] + target_directions[2]) / 2,  # Between [-1, -1] and [-1, 1]
    (target_directions[2] + target_directions[0]) / 2   # Between [-1, 1] and [1, 1]
])

# Combine the original directions with the intermediate directions
all_directions = np.vstack([target_directions, intermediate_directions])

# Normalize the directions
all_directions = all_directions / np.linalg.norm(all_directions, axis=1, keepdims=True)


R_t = 0
old_a_input = 0

total_rewards = []
episode_rewards = []

n_episodes = 1000
for i in range(n_episodes):
    state, info = env.reset()
    action = 1
    done = False
    t = 0
    total_reward = 0
    print(i)
    while not done: # and t<=max(time_steps)
        # learning_rate = max(learning_rate_min, learning_rate * learning_rate_decay)
        learning_rate = 0.01
        t+= biological_time
        action = select_action(state, weights)
        current_position_l = np.array([state[0], state[1]])
        # Take a step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        total_reward+=reward
        if done:
            print("Finished")
            print(f"rewards: {total_reward}")
            break

        # Series of six computations
        #  (1). The desired direction of
        #  cursor movement y*(t) was computed as the difference between the target
        #  position l*(t) and the current cursor position l(t). By convention, the
        #  desired direction y*(t) had unit Euclidean norm.
        y_star = target_position_l - current_position_l
        # Normalize y_star (Desired direction)
        y_star = y_star / np.linalg.norm(y_star)

        # (2) From the desired movement direction y*(t),
        # the activities x1(t), . . . ,xm(t) of the neurons
        # that provide input to the motorcortex neurons were computed via a fixed
        # linear mapping. Details on how this mapping was determined are given
        # below in Simulation details, Determination of the input activities
        x_activities = get_x_neuron_activity(y_star, weights, n_input, d)

        # (3) These input activities x were then used to calculate
        # the total synaptic activities a1(t),...,antotal
        # (t) and the resultant motor unit activities s1(t),...,sntotal (t) via
        # Equations 2 and 3 above.
        a_input = get_a_synaptic_input(weights, x_activities)
        s_activities = get_s_neuron_activity(a_input)
       

        # (4) The activities s1(t),...,sn(t) of the subset of
        # modeled recorded neurons were used to determine the cursor velocity via
        # their population activity vector,described in Equation 9 below in Simulation
        # details, Generating cursor movements from neural activity.
        # print(f"Y star velocity : {y_star}")
       
        # s_activities = s_activities[:4]

        # y_t = get_cursor_velocity(s_activities, all_directions)
        # print(f"y_t: {y_t}")
        y_t = [state[2], state[3]]
        penalty_threshold = 0.04  # Threshold for penalty

        # Check the absolute values of state[3] and state[4]
        if abs(state[2]) > penalty_threshold or abs(state[3]) > penalty_threshold:
            reward -= (abs(state[3]) - penalty_threshold) * 10000 # Apply a penalty (you can adjust this value as needed)

        if abs(state[4]) > 0.1:
            reward -= (abs(state[4])) * 100 # Apply a penalty (you can adjust this value as needed)

        

        # (5) The synaptic
        # weights wij defined in Equation 2 were updated according to a learning
        # rule, defined by Equation 16 below in Results.
        new_R_t = r_ang(y_t, y_star, reward)
        R_hat = low_pass_filter(R_t, new_R_t)
        R_t = new_R_t

        a_input_hat = low_pass_filter(old_a_input, a_input)
        old_a_input = a_input

        a_difference = (a_input - a_input_hat)
        r_difference = (R_t - R_hat)
        # print(f"r_difference: {r_difference}")
        delta_weights = learning_rate * np.outer(x_activities, a_difference).T * r_difference
        # print(f"delta_weights: {delta_weights}")
        weights += delta_weights
        state = next_state
    episode_rewards.append(total_reward)
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_episodes + 1), episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Rewards Over Episodes")
plt.legend()
plt.grid()
plt.show()
env.close()
