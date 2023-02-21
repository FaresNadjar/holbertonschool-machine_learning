#!/usr/bin/env python3

if __name__ == '__main__':
    CustomClientServiceMDP = __import__('0-client_service_MDP').CustomClientServiceMDP
    policyIteration = __import__('2-policy_iteration').policyIteration
    import numpy as np
    # define an MDP instance
    MDP = CustomClientServiceMDP()
    policy, V = policyIteration(MDP)
    print(policy)
    print(V) 
    action_names = ["charge", "support"]
    print([action_names[i] for i in policy])