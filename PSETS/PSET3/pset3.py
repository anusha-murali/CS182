# -*- coding: utf-8 -*-
"""
CS 182 Problem Set 3: Python Coding Questions - Fall 2023
Due November 15, 2023 at 11:59pm
"""

### Package Imports ###
import numpy as np
import matplotlib.pyplot as plt
### Package Imports ###

#### Coding Problem Set General Instructions - PLEASE READ ####
# 1. Unlike previous psets, this code does not need to be submitted; there is no autograder
# 2. This code goes with Problem 2: Employment Status, on this pset
# 3. This starter code has been provided to you, feel free to use it (or not, if you want to code something different) however you
#    see fit. Change variables, solve the question in another way, however you need to best understand the question and your code.
#    This coding problem can be written in a variety of different ways, this code is only a rough sketch of what your code could
#    look like. We encourage you to change it if you need to.
# 4. Make sure you write the optimal policies your code determines and copy your code and graphs onto your written submission


n_states = 3 # 0 is Safely Employed (SE), 1 is PIP, 2 is Unemployed (UE)
n_actions = 2 # 0 is Code, 1 is Netflix

t_p = np.zeros((n_states, n_actions, n_states))
# Transition Probabilities: These are represented as a 3-dimensional array
# t_p[s_1, a, s_2] = p indicates that beginning from state s_1 and taking action a will result in state s_2 with probability p
t_p[0, 0, 0] = 1        # t_p[SE, Code, SE]      = 1
t_p[0, 1, 0] = 1 / 4    # t_p[SE, Netflix, SE]   = 1 / 4
t_p[0, 1, 1] = 3 / 4    # t_p[SE, Code, PIP]     = 3 / 4  ?? t_p[SE, NetFlix, PIP] = 3 / 4
t_p[1, 0, 0] = 1 / 4    # t_p[PIP, Code, SE]     = 1 / 4
t_p[1, 0, 1] = 3 / 4    # t_p[PIP, Code, PIP]    = 3 / 4
t_p[1, 1, 1] = 7 / 8    # t_p[PIP, Netflix, PIP] = 7 / 8
t_p[1, 1, 2] = 1 / 8    # t_p[PIP, Netflix, UE]  = 1 / 8
# all other transition probabilities are 0

r = np.zeros((n_states, n_actions))
# Reward values: These are represented as a 2-dimensional array
# r[s, a] = val indicates that taking at state s, taking action a will give a reward of val
r[0, 0] = 4     # r[SE, Code]       = 4
r[0, 1] = 10    # r[SE, Netflix]    = 10
r[1, 0] = 4     # r[PIP, Code]      = 4
r[1, 1] = 10    # r[PIP, Netflix]   = 10

# Compute the utility of the given state (curState). For each state that can be reached from curState,
# we add the product of its utility, gamma and the probability of reaching that state to the reward.
def computeUtil(curState, action, gamma, modelUtils, reward):
    util = reward[curState, action]
    for s in range(n_states):
        util += t_p[curState, action, s]*gamma*modelUtils[s]
    return util

# Apply the current policy and return the converged utility values
# of the states
def policy_evaluation(policy, modelUtils, theta, gamma, reward):
    curUtil = np.zeros(n_states,dtype=float)
    while True:
        error = 0.0
        for state in range(n_states):
            curUtil[state] = computeUtil(state, policy[state], gamma, modelUtils, reward)
            error = max(error, abs(curUtil[state] - modelUtils[state]))
        modelUtils = curUtil
        if error < theta*(1-gamma) / gamma:
            break
    return modelUtils


def policy_iteration(gamma):
    """
    You should find the optimal policy for Liz under the constrants of discount factor gamma, which is given as a parameter.
    Relevant variables and the transition probabilities are defined above, feel free to use them and change them how you want.
    What this function returns is up to you and how you want to determine the sum of utilities at each iteration in the plots.
    """
    # Utility values of the states
    modelUtils = np.zeros(n_states, dtype=float)
    
    theta = 1e-5 # define a theta that determines if the change in utilities from iteration to iteration is "small enough"
    
    policy = np.zeros(n_states, dtype=int) # define your policy, which begins as Netflix regardless of state

    # Initial policy is NetFlix for all 3 states
    for i in range(len(policy)):
        policy[i] = 1

    while True:
        # Policy Evaluation
        modelUtils = policy_evaluation(policy, modelUtils, theta, gamma, r)
 
        # Policy Change check
        policy_stable = True
        
        # Policy Iteration
        for s in range(n_states):
            bestAction = None
            maxUtil = -float("inf")
            for action in range(n_actions):
                u = computeUtil(s, action, gamma, modelUtils, r)
      
                if u > maxUtil:
                    bestAction, maxUtil = action, u
            if maxUtil > modelUtils[s]: 
                policy[s] = bestAction
                policy_stable = False

        # Determine if policy has changed between iterations
        if policy_stable:
            break
    print(policy)
    return policy
        

def value_plots():
    """
    Your plots should indicate the cumulative utility summed across all states across iterations. More specifically, your y-val
    should indicate the total amount of utility acumulated across the states and actions as the iterations progress. This means
    you likely will have to keep track of what policies you have at every iteration, or some other method that will allow you to
    determine the cumulative sum of utilities as iterations continue.
    """
    
    iterations = range(0, 50)
    
    # Helper function to generate cumulative utilities
    # for a given policy. We use numpy.random.uniform 
    # to generate the desired probabilities
    def policyUtils(policy):
        policy_vals = []
        curState = 0
        cumUtils = 0.0
        for i in iterations:
            action = policy[curState]
            cumUtils = cumUtils + r[curState, action]
            policy_vals.append(cumUtils)
            # Probability for next action
            p = np.random.uniform(0, 1)
            if (curState == 0 and action == 0):
                curState = 0  # Continue to stay in SE
            elif (curState == 0 and action == 1):
                if (p > 0.25): 
                    curState = 1 # Move to PIP
            elif (curState == 1 and action == 0):
                if (p <= 0.25):
                    curState = 0 # Move to SE
            elif (curState == 1 and action == 1):
                if (p <= 1/8):
                    curState = 2 # Move to Unemployed
        return policy_vals
    
    # you will need to find a way to calculate the cumulative utility values for policy 1 and policy 2
    plt.plot(iterations, policyUtils(policy1), label="Policies for gamma = 0.9")
    plt.plot(iterations, policyUtils(policy2), label="Policies for gamma = 0.8")
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Utility Value")
    plt.legend()
    plt.title("Cumulative Utility Values over Time")
    plt.show()

if __name__ == "__main__":
    policy1 = policy_iteration(0.9) # policy iteration to verify your answer from problem 2 part c, with gamma = 0.9
    policy2 = policy_iteration(0.8) # policy iteration for problem 2 part d, with gamma = 0.8
    
    value_plots()
    # You will need to find some way to get the total utility values to the value_plots function. For example, you could pass
    # in the policies or you could pass in the cumulative sum of utility values.
