import gymnasium as gym
import matplotlib
import numpy as np

matplotlib.use('Agg')  # Use a backend suitable for your environment
import matplotlib.pyplot as plt
#
#  Simulation
#  Network model. Our aim was to explain the experimentally observed
#  learning effects in the simplest possible model. This network model con
# sisted of two populationsofneuronsconnectedinafeedforwardmanner
#  (Fig. 1B).Thefirstpopulationmodeledthoseneuronsthatprovideinput
#  to the neurons in motor cortex. It consisted of m 100 neurons with
#  activities x1(t), . . . ,xm(t)
# . Thesecondpopulationmodeledneurons
#  in motor cortex that receive inputs from the input population. It con
# sisted of ntotal
# 340 neurons with activities s1(t), . . . ,sntotal
#  (t). The dis
# tinctionbetweenthesetwolayersispurelyfunctional:inputneuronsmay
#  besituated in extracortical areas, in other cortical areas, or even in motor
#  cortex itself. The important functional feature of these two populations
#  in our model is that learning takes place solely in the synapses of projec
# tions between these populations. In principle, the same learning is appli
# cable to multilayer networks. All of the modeled motor cortical neurons
#  were used to determine the monkey arm movement in our model; how
# ever, only n 40 of these (the “recorded” subset) were used for cursor
#  control.Theactivitiesofthisrecordedsubsetaredenotedinthefollowing
#  as s1(t), . . . ,sn(t). The arm movement, based on the total population of
#  modeled motor cortex neurons, was used to determine the PDs of mod
# eled recorded neurons.

class Network():

    def __init__(self):
        self.input = 8
        self.output = 4
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        pass

    def x_activities(self, input):
        pass

    def a_synapses(self):
        pass

    def s_activities(self):
        pass

def get_s_neuron_activity(a_input):
    """Compute neuron activities based on input signals."""
    return np.maximum(0, a_input)


def get_a_synaptic_input(w, x):
    """Compute synaptic input to motor cortext neurons. With noise added to the weights"""
    return np.dot(w, x)  + get_noise()


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
    return np.random.normal(0, np.sqrt(v))

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


env = gym.make("LunarLander-v2")
states = 8
actions = 4

# Simulation params
max_t = 3000
# ,a timestep in our simulation corresponded to 1/30s in biological time.
biological_time = 1 / 30
time_steps = np.arange(0, max_t * biological_time, biological_time)
learning_rate = 0.1
v = 1  # Variance for the noise distribution
# print(f"Observation space {env.observation_space}")
n_input = states  # Observation space.
n_total = actions
d = 2  # movement dimensionality
weights = np.random.rand(n_input, n_total) * 0.001

total_rewards = []
episode_rewards = []

n_episodes = 10000
for i in range(n_episodes):
    state, obs = env.reset()
    action = 1
    done = False
    t = 0
    episode_reward = 0
    while not done:
        learning_rate = 0.01
        t += biological_time
        action = select_action(obs, weights)

        # Take a step in the environment
        next_obs, env_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # (2) From the desired movement direction y*(t),
        # the activities x1(t), . . . ,xm(t) of the neurons
        # that provide input to the motorcortex neurons were computed via a fixed
        # linear mapping. Details on how this mapping was determined are given
        # below in Simulation details, Determination of the input activities
        x_activities = get_x_neuron_activity(None, weights, n_input, d)

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


        # (5) The synaptic
        # weights wij defined in Equation 2 were updated according to a learning
        # rule, defined by Equation 16 below in Results.

        # print(f"r_difference: {r_difference}")
        delta_weights = learning_rate * np.outer(x_activities, a_difference).T * r_difference
        # print(f"delta_weights: {delta_weights}")
        weights += delta_weights
        obs = next_obs
    print("Finished")
    print(f"rewards: {episode_reward}")
    episode_rewards.append(episode_reward)
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_episodes + 1), episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Rewards Over Episodes")
plt.legend()
plt.grid()
plt.savefig("reward_plot.png")
env.close()

# noisy neuronal output is used for exploration to improve performance
# does not require any external in formation todifferentiate internal noise from synaptic input