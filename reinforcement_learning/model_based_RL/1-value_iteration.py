#!/usr/bin/env python3
"""
    Value Iteration Algo
"""
import numpy as np


def valueIteration(CustomMDP, gamma=0.9):
    """
        Performs Value iteration algorithm
    """

    probs = CustomMDP.probs
    n_actions = CustomMDP.actions
    n_states = CustomMDP.states
    # initialize V
    V = [0, 0, 0, 0]
    policy = [0, 0, 0, 0]
    V_old = [-1, -1, -1, -1] # estimate from the previous step

    while (abs(np.array(V) - np.array(V_old))).max() > 1e-3:
      V_old = V.copy()
      for s in range(n_states):
        # for every s we have q: q is the variable of the possible results of this summation
        q = [0, 0]
        for a in range(n_actions):
          
          # for s_next in range(states):
          #   q[a] += probs[s, a, s_next] * (R[s, a, s_next] + gamma * V[s])
          for s_next, prob_reward in enumerate(probs[s][a]):
            # prob_reward = (trasition probability of state s_next, reward of total path stat -> action -> next state)
            q[a] += prob_reward[0] * (prob_reward[1] + gamma * V_old[s_next])
        # print(q)
        V[s] = max(q)
        if s == 0:
            # the only action is 1: support so pass this step
            policy[s] = 1
        else:
          policy[s] = q.index(max(q))

    return policy, V
