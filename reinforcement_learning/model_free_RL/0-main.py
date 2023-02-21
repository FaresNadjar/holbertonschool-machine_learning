#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import numpy as np
    customEnv = __import__('0-custom_env').customEnv

    # Possible states of the environment
    # 0: No item in the stock, 1: one item in the stock 2: 2 items in the stock (Full)
    possible_states = [0,1,2] # seen as the vertices of the graph
    # Define the possible action for each state, [index 0]: none, [index 1] restock
    possible_actions = { 0 : [0,1], 1 : [1,2], 2:[2]}  # seen as the edges of the graph

    # Define the possible events for each state and  the next state caused by an event
    # Possible events, [index 0]: sell, [index 1] no sell
    possible_events  = {0 : [0,0], 1 :[1,0] , 2 :[2,1]}
    # Define the events probabilities for each state
    prob_events      = {0 : [.3,.7], 1 :[.3,.7] , 2 :[.3,.7]}

    # The policy is to push to more sells and restock action
    # Rewards are based on the (state, index of the event) : value of the reward
    rewards = {(0,1) : -0.2,
              (0,0) : 0,
              (1,1) : 1,
              (1,0) : 0,
              (2,1) : 1,
              (2,0) : -0.5}  


    # define the inital state odf the environment (the stock of items)
    init_state = 1
    # ENV instance
    env = customEnv(init_state, possible_actions, possible_events, rewards, prob_events)
    print("Environment instance type", type(env))

    print("The initial state is", env.state )
    print("the possible actions for the environment are", env.possible_actions)
    print("The possible events that could occur in the environment are", env.possible_events)
    print("The agent's rewards are as follows", env.rewards )
    print("The probabilities for each event", env.prob_events)