#!/usr/bin/env python3
"""
    Policy Iteration Algo
"""
import numpy as np


def policyIteration(CustomMDP, gamma=0.9, delta = 0.2):
    """
        Performs Policy iteration algorithm
    """
    
    probs = CustomMDP.probs
    n_actions = CustomMDP.actions
    n_states = CustomMDP.states
    # Set policy iteration parameters
    max_policy_iter = 10  # Maximum number of policy iterations
    max_value_iter = 10 # Maximum number of value iterations
    pi = [0 for s in range(n_states)]
    V = [0 for s in range(n_states)]


    for i in range(max_policy_iter):
        # Initial assumption: policy is stable
        optimal_policy_found = True
        # Policy evaluation
        # Compute value for each state under current policy
        for j in range(max_value_iter):
            max_diff = 0  # Initialize max difference
            V_new = [0, 0, 0, 0, 0]  # Initialize values
            for s in range(n_states):
                # Compute state value
                val = 0
                for s_next, prob_reward in enumerate(probs[s][pi[s]]):
                    val += prob_reward[0] * (prob_reward[1] + gamma * V[s_next])


                # Update maximum difference
                max_diff = max(max_diff, abs(val - V[s]))

                V[s] = val  # Update value with highest value
            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break

        # Policy iteration
        # With updated state values, improve policy if needed
        for s in range(n_states):

            val_max = V[s]
            for a in range(n_actions):
                # Compute state value
                val = 0
                for s_next, prob_reward in enumerate(probs[s][a]):
                    val += prob_reward[0] * (prob_reward[1] + gamma * V[s_next])

                # Update policy if (i) action improves value and (ii) action different from current policy
                if val > val_max and pi[s] != a:
                    pi[s] = a
                    val_max = val
                    optimal_policy_found = False

        # If policy did not change, algorithm terminates
        if optimal_policy_found:
            # force the first state policy to be 1: support
            pi[0] = 1
            break
    return pi, V