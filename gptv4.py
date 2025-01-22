import numpy as np
import gym

class EHAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-3, exploration_level=0.1):
        self.weights = np.random.uniform(-0.5, 0.5, (state_dim, action_dim))
        self.learning_rate = learning_rate
        self.exploration_level = exploration_level
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.last_reward = 0

    def act(self, state):
        noise = np.random.uniform(-self.exploration_level, self.exploration_level, self.action_dim)
        action = np.dot(state, self.weights) + noise
        return np.tanh(action), noise  # Ensure action is within [-1, 1]

    def update(self, state, action_noise, reward):
        filtered_reward = 0.8 * self.last_reward + 0.2 * reward
        for i in range(self.state_dim):
            for j in range(self.action_dim):
                state_mean = np.mean(state)
                noise_mean = np.mean(action_noise)

                # EH Rule Update
                delta_w = (
                    self.learning_rate
                    * (state[i] - state_mean)
                    * action_noise[j]
                    * (reward - filtered_reward)
                )
                self.weights[i, j] += delta_w

        self.last_reward = filtered_reward

def train_pendulum(env_name="Pendulum-v1", episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = EHAgent(state_dim, action_dim)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, noise = agent.act(state)
            next_state, reward, done, _ = env.step([action])

            agent.update(state, noise, reward)

            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train_pendulum()