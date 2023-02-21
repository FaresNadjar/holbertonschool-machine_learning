#!/usr/bin/env python3
"""
    Build the Q-learning algorithm
"""
import numpy as np
import random


def SARSA_Agent(env, Q, episodes=100, max_steps=2000, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
        Perform the SARSA
    """
    def best_poss_action(Q, state):
        """
            Take the best action
        """
        # returns the index (in the list of all possible actions in the current state)
        # of one action randomly chosen in the set of best possible actions according to Q
        # print(state)
        Q_st = Q[state]
        best=list(np.nonzero(Q_st==np.max(Q_st))[0])
        # how random is that ? we only have one item in the list best!! alwayyyyys
        return random.choice(best)   
  
    def eps_greedy(Q, state, possible_actions, epsilon):
        """
            Perform the Epsilon greedy function
        """
        r = np.random.uniform(0,1)
        if r < epsilon:
          # explore
          possible_actions = possible_actions[state]
          l=len(possible_actions)
          if l>1 : # if we are NOT in the state of full stock!
              action_index = np.random.choice([0,1])
          else: # if we are in the state of full stock!
              action_index = 0
        else:
          # exploit
          action_index = best_poss_action(Q, state)
        return action_index
    
    max_epsilon = epsilon
    # List of rewards
    rewards_all_episodes = []

    # start training
    for episode in range(episodes):
        # Reset the environment
        # SARSA - initialize s
        state = env.reset()
        # SARSA - choose a from s using policy derived from Q (policy: eps-freedy)
        action = eps_greedy(Q, state, env.possible_actions, epsilon)
        done = False
        rewards_current_episode = []
        for step in range(max_steps):
            # take action a(action), observe r, s'(state_)
            state_, reward, done, info = env.step(action)
            # choose a'(action_) from s' using policy derived from Q
            action_ = eps_greedy(Q, state_, env.possible_actions, epsilon)
            
            # we update Q
            
            target = reward + gamma * Q[state_][action_]

            Q[state][action] *= 1-alpha
            # print(self.Q)
            Q[state][action] += alpha*target
            
            rewards_current_episode.append(reward)
            # and the event finally happens
            state = state_
            action = action_

            if done is True:
                break
        # Reduce epsilon (because we need less and less exploration)
        eps = (min_epsilon + (max_epsilon - min_epsilon) *np.exp(-epsilon_decay * episode))
        rewards_all_episodes.append(rewards_current_episode)
    
    return Q, rewards_all_episodes