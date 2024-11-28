import numpy as np


def get_s_neuron_activity(a_input):
    s_activities = np.arange(0, n_recorded)
    for i in range(n_recorded):
        s_activities[i] = delta(a_input[i])
    return s_activities


def get_a_synaptic_input(w, x):
    a_input = np.arange(0, n_recorded)
    for i in range(n_recorded):
        # Generate noise for each neuron
        noise = get_noise()
        for j in range(n_input):
            a_input[i] = w[i][j] * x[j] + noise
    return a_input


def get_x_neuron_activity(y_star, weights):
    # TODO: Let Q be the 3 ntotal matrix where column i is given by qi from
    #  Equation 1 for i 1,...,ntotal.
    Q = np.zeros((3, n_total))
    Q_inverse = np.linalg.pinv(Q)
    s_tilde = np.dot(Q_inverse, y_star)
    W_total = weights  # TODO: This is wrong, should be the weights before learning
    c_rate = 1  # Scaling factor for hertz
    x_activities = c_rate * np.dot(np.linalg.pinv(W_total), s_tilde)
    return x_activities


def delta(x):
    # Treshold linear activation function that assures non-negative activities
    if x > 0: return x
    return 0


def get_noise():
    return np.random.normal(0, np.sqrt(variance))


def get_cursor_velocity(s_activities):
    k_s = 1  # Constant to convert magnitude of population vector to speed
    # TODO: calculate the values for the variables
    beta = 1
    alpha = 1
    p = 1
    total = 0
    for i in range(n_recorded):  # Not sure if this should be recored neurons or input neurons
        total += ((s_activities[i] - beta) / alpha) * p
    return k_s * d / n_input * total


def r_ang(y, y_star):
    """Angle between movement direction and desired direction"""
    r_t = (y * y_star) * np.linalg.norm(y)  # TODO: y should be transposed
    return r_t


def low_pass_filter(previous, current):
    filtered_value = 0.8 * previous + 0.2 * current
    return filtered_value


# Simulation params
max_t = 100
time_steps = np.arange(0, max_t)
learning_rate = 0.01
variance = 1  # Variance for the noise distribution
# Parameters
n_input = 100
n_total = 340
n_recorded = 40
d = 3  # movement dimensionality
# ,a timestep in our simulation corresponded to 1/30s in biological time.
weights = np.random.uniform(-0.5, 0.5, (n_total, n_input))

target_position_l = np.random.choice([-1, 1], size=3)
current_position_l = np.zeros(3)
# Eigth corners of a cube
target_directions = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                              [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])
target_directions = target_directions / np.linalg.norm(target_directions, axis=1, keepdims=True)
R_t = 0
old_a_input = 0
for t in time_steps:
    # Generate noise at each timestep from zero mean distribution:

    # TODO: determine x_activities
    # Determine activities of neurons s_i(t) as delta(a_i(t))

    # Series of six computations
    #  (1). The desired direction of
    #  cursor movement y*(t) was computed as the difference between the target
    #  position l*(t) and the current cursor position l(t). By convention, the
    #  desired direction y*(t) had unit Euclidean norm.
    y_star = target_position_l - current_position_l
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
    y_t = get_cursor_velocity(s_activities)
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
    r_difference = (R_t - R_hat)
    print(f"len a {len(a_difference)}")
    print(f"len r {len(r_difference)}")
    delta_weights = learning_rate * np.dot(np.dot(x_activities, a_difference), r_difference)
    print(delta_weights)
    weights += delta_weights
    # 6) Finally, if the new  cursor location was close to the target (i.e., if l(t) l*(t)  0.05),
    # we deemed it a hit, and the trial ended. Otherwise, we simulated another
    # time step and returned to computation step 1. In summary, every trial
    # was simulated as follows:
