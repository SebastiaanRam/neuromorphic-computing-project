import numpy as np

d = 3  # movement dimensionality
# First population: neurons that provide input to the neurons in motor cortex. I
m = 100  # neurons
x_activities = np.arange(0, m)  # x_1(t) to ... x_m(t)
# Second population: modeled neurons in motor cortex that receive inputs from the input population
n_total = 340  # neurons
s_activities = np.arange(0, n_total)  # s_1(t) to ... s_n(t)

n = 40  # number of recorded neurons

# Values to be determined
k = 1  # converts the magnitude of the popultation vector to speed


def y_arm(t, q_i):
    """
    Direction of the monkey arm movement at time t.
    I think this should return a vector aswel
    :param t: Time point
    :param q_i: 3d-Vector with direction in which neuron i contributes to the movement
    :return:
    """
    total = 0
    for i in range(n_total):
        total += s_activities[i](t) * q_i
    return total


# y_star: Desired movement direction

def D(v):
    """
    Zero mean distribution
    :param v: Determines the variance (amount of noise in the neuron
    :return: Sample from the distribution
    """
    mean = 0  # Zero mean distribution
    standard_deviation = np.sqrt(v)
    return np.random.normal(mean, standard_deviation)


def epsilon(t):
    """
    Generates a sample from D(v)
    :param t: Time step
    :return: Sample from D(v)
    """
    variance = 2  # set to two for now, can be changed
    return D(variance)


def a(t, i, w):
    total = 0
    for j in range(m):
        total += w[i][j] * x(t, j) + epsilon(t)
    return total


def s(t, i, w):
    """
    Activity of neuron i at time t

    :param t: Time step
    :param i: Neuron i
    :return: activity of the neuron at the specified time
    """
    return delta(a(t, i, w))


def delta(x):
    """Linear activation function"""
    if x > 0:
        return x
    return 0


def q_values():
    """
    Determination of input activities
    :return:
    """
    q_1 = np.arange(0, n_total)
    q_2 = np.arange(0, n_total)
    phi = np.random.uniform(0, 2 * np.pi, size=n_total)
    q_3 = np.random.uniform(-1, 1, size=n_total)
    for i in range(n_total):
        q_1[i] = np.sqrt(1 - q_3[i] ** 2) * np.cos(phi[i])
        q_2[i] = np.sqrt(1 - q_3[i] ** 2) * np.sin(phi[i])
    return q_1, q_2, q_3


Q = np.random.uniform(-1, 1, size=(3, n_total))  # Q matrix TODO: Column i should be given by q_i
Q_pseudo_inv = np.linalg.pinv(Q)
s_tilde = lambda y_star: Q_pseudo_inv @ y_star
W_total = 1
# Let Wtotal denote the matrix of weights wij before learning,
# i.e., the element of Wtotal in row i and column j is the weight from input
# neuron j to neuron i in the simulated motor cortex before learning.

beta = np.arange(0, n)  # Beta values to be determined via linear regression
alpha = np.arange(0, n)  # same as beta
p = np.arange(0, n)  # same as beta


def x(t, y_star):
    """
    Activities of the input neurons.

    :param t: Time step
    :param y_star: Desired direction
    :return:
    """
    c_rate = 0.5  # Scaling factor to scale the input activity to convert to hertz (
    x_t = c_rate * np.linalg.pinv(W_total) * Q_pseudo_inv * y_star(t)
    return x_t


def y(t, w):
    """
    Generating movements from neural activity.

    :param t:
    :return:
    """
    d = 3  # Movement dimensionality
    k_s = 1 / 30  # converts magnitude of the population vector to speed (unsure what it is)
    total = 0
    for i in range(n):
        total += ((s(t, i, w) - beta[i]) / alpha[i]) * p[i]
    y_t = k_s * d / n * total
    return y_t


def cursor_position(y, l_0, t_steps, dt):
    """
    Compute cursor position l(t) by integrating velocity signal y(t).

    :param y: Function that computes velocity y(t) at a given time t.
    :param l_0: Initial position vector (e.g., np.array of shape (d,)).
    :param t_steps: Number of time steps to simulate.
    :param dt: Time step size (Δt).
    :return: Array of cursor positions l(t) at each time step.
    """
    d = len(l_0)  # Dimensionality of the space
    l = np.zeros((t_steps, d))  # Store cursor positions over time
    l[0] = l_0  # Set initial position

    for t in range(1, t_steps):
        current_time = t * dt
        l[t] = l[t - 1] + dt * y(current_time)  # Update position

    return l


def R_ang(t, y_star, w):
    return y(t, w) * y_star(t) / np.linalg.norm(y(t, w), w)


def R_batch(T, y_star, w):
    total = 0
    for t in range(T):
        total += R_ang(t, y_star, w)
    R_batch = 1 / T * total
    return R_batch


def w(t, i, j, w, learning_rate=0.1):
    y_star = [1, 1, 1]
    delta_w = learning_rate * x(t, j) * [a(t, i, w) - a(t, i, w)] * [R_ang(t, y_star, w) - R_ang(t, y_star, w)]
    return delta_w
