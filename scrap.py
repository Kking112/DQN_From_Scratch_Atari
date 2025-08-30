import gymnasium as gym
import ale_py
import numpy as np
# register ale-py with gym, ale-py is the library that has all of the Atari games
gym.register_envs(ale_py)

env = gym.make('SpaceInvadersNoFrameskip-v4')

env.observation_space._shape

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(obs)
# print env attributes to get a sense of what we're working with
print("Observation shape: ", obs.shape) # printing the shape b/c it's a 3D array
print("Action space: ", env.action_space)
print("Reward: ", reward)
print("Terminated: ", terminated)
print("Truncated: ", truncated)
print("Info: ", info)

# * Prints:
# Observation shape:  (210, 160, 3)
# Action space:  Discrete(6)
# Reward:  0.0
# Terminated:  False
# Truncated:  False
# Info:  {'lives': 3, 'episode_frame_number': 1, 'frame_number': 1}

# print dtypes
print("Observation dtype: ", env.observation_space.dtype)
print("Action dtype: ", env.action_space.dtype)
print("Reward dtype: ", type(reward))
print("Terminated dtype: ", type(terminated))
print("Truncated dtype: ", type(truncated))
print("Info dtype: ", type(info))
# * Prints:
# Observation dtype:  uint8
# Action dtype:  int64
# Reward dtype:  <class 'float'>
# Terminated dtype:  <class 'bool'>
# Truncated dtype:  <class 'bool'>
# Info dtype:  <class 'dict'>



### ! Tuple test
tup2 = (64,)
tup1 = (24,24,24)

tup_3 = tup1 + tup2
