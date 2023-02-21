#!/usr/bin/env python3
"""
    custom ENV class 
"""
import random


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

    def step(self, action_index):
        """
        Method: Take a sep into the ENV giving an action index

        Returns: next_state of the ENV,
                 reward taken in that step,
                 done true or false, info
        """
        info = {}
        done = False
        # the old state of the current step
        old_state = self.state
        # print("old sate", old_state)
        # possible actions according to the current: considered old state
        poss_states = self.possible_actions[old_state] # --> [0, 1] or [1, 2] or [2] those are the future states given a state index


        if old_state == 2: # if the state is 2 no option for action_index to be other than 0 (yeah like we are cheating or what! / or just it is about the dynamics, but stll no penalty)
            action_index = 0 # the dynamics of the Env say so!
        # now change env sate based on the action taken!
        # current env state based on the action taken!
        curr_env_state = poss_states[action_index]
        # let's observe more
        # then an event will happen. Let us observe which one (among the possible ones in the current state)
        poss_ev = self.possible_events[curr_env_state]
        # let's chack the probability of these posiible events of the current env state
        probs   = self.prob_events[curr_env_state]
        # get the index of the event that happens based on a probability distribution of total events (2 events: sell or not sell)
        l=len(poss_ev)
        event_index = random.choices(range(l),weights=probs,k=1)[0]
        # print("event index(0: no selling, 1: selling) =", event_index)
        # end of event!

        # no more changes let's get our reward!
        # we see what will be the reward: given a current state of the stock and event!
        reward = self.rewards[(curr_env_state, event_index)]

        # then the real change!
        # this event will chage the current env state and give us the next step state!
        next_state = poss_ev[event_index]
       
        info["event_index"] = event_index

        # we are missing the Done concept! (what ends an episode other than reaching the final step)
        # done = "done"
        self.state = next_state
        return next_state, reward, done, info
