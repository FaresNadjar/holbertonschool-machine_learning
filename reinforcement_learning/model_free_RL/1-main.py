#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import numpy as np
    customEnv = __import__('1-custom_env').customEnv

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
    print("Taking a step with action index 0 \"Do nothing\"")
    # take a step and make decision taking the action with index 0
    next_state, reward, done, info = env.step(0)
    # print the environment event
    if info['event_index'] == 0:
        print("\"No sales\" have occurred as a result of the environment's event.")
    if info['event_index'] == 1:
        print("There was \"a selling\" event that took place.")
    # print the next state of the environment
    print("The Next state index is ", next_state)

