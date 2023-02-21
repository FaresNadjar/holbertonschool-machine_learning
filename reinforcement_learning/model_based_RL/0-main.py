#!/usr/bin/env python3

if __name__ == '__main__':
    CustomClientServiceMDP = __import__('0-client_service_MDP').CustomClientServiceMDP
    # define an MDP instance
    MDP = CustomClientServiceMDP()
    print(type(MDP))
    action_space = MDP.actions
    print("Action space dimension = ", action_space)
    state_space = MDP.states
    print("State space dimension = ", state_space)
    transition_probabilities = MDP.probs
    print("Transition probabilities = ", transition_probabilities)