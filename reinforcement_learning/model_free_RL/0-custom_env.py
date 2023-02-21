#!/usr/bin/env python3
"""
    custom ENV class 
"""


class customEnv():
    """
        class env 
    """
    def __init__(self,init_state,
                    possible_actions,
                    possible_events,
                    rewards, prob_events):
        """
            class constructor
        """
        self.state            = init_state
        self.possible_actions = possible_actions
        self.possible_events  = possible_events
        self.prob_events      = prob_events
        self.rewards          = rewards
