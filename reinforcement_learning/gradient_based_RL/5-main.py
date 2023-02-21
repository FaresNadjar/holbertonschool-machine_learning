#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import random

    customEnv = __import__("2-custom_env").customEnv
    simulate_one_step = __import__("5-simulation").simulate_one_step

    tf.random.set_seed(0)
    random.seed(0)

    # Possible states of the environment
    # 0: No item in the stock, 1: one item in the stock 2: 2 items in the stock (Full)
    possible_states = [0,1,2] # seen as the vertices of the graph
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

    init_state = 0
    env = customEnv(init_state, possible_actions, possible_events, rewards, prob_events)
    # render initial state of the environment
    print(env.render())
    # the inital state of the environment is the step state
    step_state = init_state
    
   
    # take the model path
    model_path = "./model"
    # do simulation for one step
    action_taken, new_state, reward, info = simulate_one_step(env, step_state, model_path)
    
    if step_state == 0:
        print("The environment is in the state that \"there are no items in the stock\".")
    if step_state == 1:
        print("The environment is in the state that \"there is one item in the stock\".")
    if step_state == 2:
        print("The environment is in the state that \"there are two items in the stock\".")
    
    
    # print the action taken
    if action_taken == 0:
      print("The agent's response is to \"do nothing\".")
    if action_taken == 1:
      print("The agent's response is to \"Restock\".")

    # print the environment event
    if info['event_index'] == 0:
        print("\"No sales\" have occurred as a result of the environment's event.")
    if info['event_index'] == 1:
        print("There was \"a selling\" event that took place.")

    # print the new state of the environment or render
    if new_state == 0:
        print("The environment is in the state that \"there are no items in the stock\".")
    if new_state == 1:
        print("The environment is in the state that \"there is one item in the stock\".")
    if new_state == 2:
        print("The environment is in the state that \"there are two items in the stock\".")