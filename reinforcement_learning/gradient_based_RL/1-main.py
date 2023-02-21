#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import tensorflow as tf
    import random
    build_NNagent = __import__('0-build_NN_agent').build_NNagent
    get_action = __import__('1-tensorflow_probability').get_action
    
    tf.random.set_seed(0)
    random.seed(0)

    model = build_NNagent()
    state = [1, 0, 0]

    distribution, action_index = get_action(model, state)
    print("Distribution of action for index 0 is ", distribution.prob(0))
    print("Distribution of action for index 1 is ", distribution.prob(1))


    print("action index sampled is ", action_index)