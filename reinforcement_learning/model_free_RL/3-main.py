#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import numpy as np
    customEnv = __import__('2-custom_env').customEnv
    qLearningAgent = __import__('3-q_learning_agent').qLearningAgent
    import matplotlib.pyplot as plt
    # customEnv = __import__('0-custom_environment').customEnv
    # qLearningAgent = __import__('1-q_learning_agent').qLearningAgent
    
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
    
    
    # define the inital state of the environment (the stock of items)
    init_state = 1
    # ENV instance
    env = customEnv(init_state, possible_actions, possible_events, rewards, prob_events)
    # initialize the Q-table
    q_table = {x:np.array([0.0 for y in possible_actions[x]]) for x in possible_states}
    print("initial Q-table: ", q_table)
    print("----------------------------")
    print("Start the training")
    Q, rewards_all_episodes = qLearningAgent(env, q_table)
    print("Q-table after training: ", Q)
    # f = open('results.txt', 'a')
    # f.write(str(Q) + "\n")
    # f.close()
    
    print("Use the Q-table and make a step")
    # define the inital state
    init_state = 2
    # define a new environment instance
    env_s = customEnv(init_state, possible_actions, possible_events, rewards, prob_events)
    print("Simulating the Environment")
    print("--------------------------")
    """
		Please run the code several times as the next state is determined by the dynamics of the environment step, which involves a random event (sale or no sale),

		You are free to try a different initial state each time!
    """

    print("Render the environment")
    env_s.render()
    print("taking a step")
    # take a step and make decision based on the q-table values
    next_state, reward, done, info = env_s.step(np.argmax(q_table[init_state]))

    # print the environment event
    if info['event_index'] == 0:
        print("\"No sales\" have occurred as a result of the environment's event.")
    if info['event_index'] == 1:
        print("There was \"a selling\" event that took place.")

    
    print("Render the environment after taking a step")
    env_s.render()

    # # print("plot training rewards")
    # for rewards_current_episode in rewards_all_episodes:
    #     plt.plot(np.array(rewards_current_episode).cumsum())
    # plt.show()
