#!/usr/bin/env python3

if __name__ == '__main__':
    CustomClientServiceMDP = __import__('0-client_service_MDP').CustomClientServiceMDP
    valueIteration = __import__('1-value_iteration').valueIteration
    import numpy as np
    # define an MDP instance
    MDP = CustomClientServiceMDP()
    policy, V = valueIteration(MDP)
    print(policy)
    print(V) 
    action_names = ["charge", "support"]
    print([action_names[i] for i in policy])