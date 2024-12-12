import gym
import numpy as np


def get_s_neuron_activity(a_input):
    """Compute neuron activities based on input signals."""
    # I think np.maximum(0, a_input) should have the same result????
    s_activities = np.arange(0, n_recorded)
    for i in range(n_recorded):
        s_activities[i] = delta(a_input[i])
    return s_activities


def get_a_synaptic_input(w, x):
    """Compute synaptic input to motor cortext neurons. With noise added to the weights"""
    # a_input = np.arange(0, n_recorded)
    # for i in range(n_recorded):
    #     # Generate noise for each neuron
    #     noise = get_noise()
    #     for j in range(n_input):
    #         a_input[i] = w[i][j] * x[j] + noise
    return np.dot(w, x) + get_noise()


def get_x_neuron_activity(y_star, weights):
    # Ensure the desired movement direction is normalized
    y_star = y_star / np.linalg.norm(y_star)
    
    # Generate a random matrix Q representing the mapping of y_star to s_neuron activities - Neural connectivity weights
    #  the range [−1,1] is a reasonable, biologically plausible, and computationally safe choice for initializing 𝑄. It balances randomness, symmetry, and numerical stability.
    Q = np.random.uniform(-1, 1, (d, n_total))
    Q_inverse = np.linalg.pinv(Q)  # Invert Q to map back from s_neuron to input activities
    
    # Compute intermediate target activities (s_tilde) in the motor cortex
    s_tilde = np.dot(Q_inverse, y_star)
    
    # Scale input neuron activities to reflect pre-synaptic neuron population firing rates
    c_rate = 10  # Scaling factor for converting normalized activities to firing rates in Hz
    #  Input neuron activities (x_activities) based on a scaled inverse mapping of the target motor cortex activities (s_tilde) using the weight matrix (weights). 
    x_activities = c_rate * np.dot(np.linalg.pinv(weights), s_tilde)
    
    # Apply a threshold-linear activation to ensure non-negative activities, maybe apply here instead of delta?
    # x_activities = np.maximum(0, x_activities)
    return x_activities


def delta(x):
    """Threshold-linear activation function that ensures non-negative activities."""
    return max(0, x)


def get_noise():
    """Generate noise from a zero-mean distribution."""
    return np.random.normal(0, np.sqrt(variance))


def get_cursor_velocity(s_activities):
    # TODO: calculate the values for the variables
    k_s = 1  # Constant to convert magnitude of population vector to speed
    alpha = 1
    beta = 1
    p = 1  # Scaling factor
    scaled_activities = ((s_activities - beta) / alpha) * p
    return k_s * d / n_input * np.sum(scaled_activities)
    # k_s = 1  # Constant to convert magnitude of population vector to speed
    # # TODO: calculate the values for the variables
    # beta = 1
    # alpha = 1
    # p = 1
    # total = 0
    # for i in range(n_recorded):  # Not sure if this should be recored neurons or input neurons
    #     total += ((s_activities[i] - beta) / alpha) * p
    # return k_s * d / n_input * total


def reward(y, y_star):
    # Y: current location
    # y_star: Desired direction
    pass


def r_ang(y, y_star):
    """Angle between movement direction and desired direction"""
    print(f"Y: {y}")
    print(f"Y_star: {y_star}")
    r_t = np.dot(y, y_star) * np.linalg.norm(y)  # TODO: y should be transposed
    print(f"R_ang {r_t}")
    return r_t


def low_pass_filter(previous, current, alpha=0.2):
    """Low pass filter to smooth values"""
    return (1 - alpha) * previous + alpha * current


# Simulation params
max_t = 100
# ,a timestep in our simulation corresponded to 1/30s in biological time.
biological_time = 1 / 30
time_steps = np.arange(0, max_t * biological_time, biological_time)
learning_rate = 0.01
variance = 1  # Variance for the noise distribution
# Parameters
n_input = 100
n_total = 340
n_recorded = 40
d = 2  # movement dimensionality
weights = np.random.uniform(-0.5, 0.5, (n_total, n_input))

target_position_l = np.random.choice([-1, 1], size=d)
current_position_l = np.zeros(d)
# 4 corners of the gridworld
target_directions = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
target_directions = target_directions / np.linalg.norm(target_directions, axis=1, keepdims=True)
R_t = 0
old_a_input = 0

total_rewards = []

# Setup environment
env = gym.make("LunarLander-v2")
# Reset environment
state, info = env.reset()

for t in time_steps:
    # Random action
    action = env.action_space.sample()
    # Take a step in the environment
    next_state, reward, done, truncated, info = env.step(action)
    total_rewards.append(reward)
    # Generate noise at each timestep from zero mean distribution:

    # TODO: determine x_activities
    # Determine activities of neurons s_i(t) as delta(a_i(t))

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
    x_activities = get_x_neuron_activity(y_star, weights)
    # (3) These input activities x were then used to calculate
    # the total synaptic activities a1(t),...,antotal
    # (t) and the resultant motor unit activities s1(t),...,sntotal (t) via
    # Equations 2 and 3 above.
    a_input = get_a_synaptic_input(weights, x_activities)
    s_activities = get_s_neuron_activity(a_input)
    print(s_activities)
    # (4) The activities s1(t),...,sn(t) of the subset of
    # modeled recorded neurons were used to determine the cursor velocity via
    # their population activity vector,described in Equation 9 below in Simulation
    # details, Generating cursor movements from neural activity.
    y_t = [get_cursor_velocity(s_activities), get_cursor_velocity(s_activities)]
    print(f"len y : {y_t}")
    # (5) The synaptic
    # weights wij defined in Equation 2 were updated according to a learning
    # rule, defined by Equation 16 below in Results.
    # EH rule
    new_R_t = r_ang(y_t, y_star)
    R_hat = low_pass_filter(R_t, new_R_t)
    R_t = new_R_t

    a_input_hat = low_pass_filter(old_a_input, a_input)
    old_a_input = a_input
    print(f"A input {a_input}")
    print(f"A input hat {a_input_hat}")
    print(f"R t {R_t}")
    print(f"R hat {R_hat}")

    a_difference = (a_input - a_input_hat)
    r_difference = (reward - np.mean(total_rewards))
    print(f"len a {len(a_difference)}")
    print(a_difference.shape)
    print(f"len r {r_difference}")
    print(f"len x {len(x_activities)}")
    # delta_weights = learning_rate * np.dot(np.dot(x_activities, a_difference), r_difference)
    delta_weights = learning_rate * np.outer(x_activities, a_difference).T * r_difference
    print(delta_weights)
    weights += delta_weights
    print(weights)
    # 6) Finally, if the new  cursor location was close to the target (i.e., if l(t) l*(t)  0.05),
    # we deemed it a hit, and the trial ended. Otherwise, we simulated another
    # time step and returned to computation step 1. In summary, every trial
    # was simulated as follows:
    if np.linalg.norm(current_position_l - target_position_l) < 0.05:  # not sure if this needs to be normalized
        print(f"Finished timestep: {t}")
        print(f"Final weights: {weights}")
        break
