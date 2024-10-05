import gym
from time import sleep
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

# env = gym.make("FrozenLake-v1", map_name="4x4", render_mode = "human", is_slippery=True) # Set up the Frozen lake environmnet 
env.reset()

print("Testing Value Iteration...")
sleep(1)
my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.01) # Instantiate class object

# print("1: ", my_policy.V)
    
my_policy.value_iteration() # Iterate to derive the final policy
