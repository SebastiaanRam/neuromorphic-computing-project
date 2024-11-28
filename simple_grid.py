import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_simplegrid

env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
obs, info = env.reset()
done = env.unwrapped.done

for _ in range(50):
    if done:
        break
    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
env.close()