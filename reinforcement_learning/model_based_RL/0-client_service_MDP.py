#!/usr/bin/env python3
"""
    Building the MDB
"""

class CustomClientServiceMDP:
    """class MDP """
    def __init__(self):
      """
      Initialize Markov Decision Process model
      """

      # actions (0 = charge, 1 = support)
      self.actions = 2
      # states (0:none, 1:low, 2:medium, 3: high)
      self.states = 4
      # transition probabilities and rewards forme is [state][action][next_state](transition probability to the next_state, reward)
      self.probs = [
          [[(0.5  , -2000), (0.5  ,   -2200), (0  ,   0), (0  ,   0)], [(0.1, -1500), (0.9, -1500), (0   ,   0), (0   , 0)]],
          [[(0.75, 0), (0.25,   0), (0  ,   0), (0  ,   0)], [(0  ,     0), (0.45,   50), (0.55 , 50), (0   , 0)]],
          [[(0  , 0), (0.8, 0), (0.2, 0), (0  ,   0)], [(0  ,     0), (0  ,     0), (0.2,   250), (0.8, 250)]],
          [[(0  , 0), (0  ,   0), (0.65, 200), (0.35, 600)], [(0  ,     0), (0  ,     0), (0.15 ,   0), (0.85 , 0)]]
      ]
      