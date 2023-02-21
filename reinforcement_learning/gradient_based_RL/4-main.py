#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import random
    customEnv = __import__("2-custom_env").customEnv
    REINFORCE = __import__("4-reinforce_algorithm").REINFORCE

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

    init_state = 1
    env = customEnv(init_state, possible_actions, possible_events, rewards, prob_events)
    path_save_model = "./model"
    total_reward_episodes = REINFORCE(env, path_save_model)

    # Check the model output after training for each state
    NNagent = tf.keras.models.load_model(path_save_model)
    state_0= np.array([[0, 0, 1]])
    prediction_0 = NNagent.predict(state_0)
    print(prediction_0)

    state_1= np.array([[0, 1, 0]])
    prediction_1 = NNagent.predict(state_1)
    print(prediction_1)

    state_2= np.array([[1, 0, 0]])
    prediction_2 = NNagent.predict(state_2)
    print(prediction_2)