import gymnasium as gym

# Initialise the environment with human-rendering mode
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # Insert your policy here
    action = env.action_space.sample()

    # Step (transition) through the environment with the action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended, reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()