#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import tensorflow as tf
    import random
    trainAgent = __import__('3-train_NN_agent').trainAgent

    random.seed(0)

    states = [[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    rewards = [1, 1, -0.2, -0.2, 0]
    actions = [1, 0, 0, 0, 1]
    path_save_model = "./testmymodel"
    trainAgent(states, rewards, actions, path_save_model, gamma=0.9, l_rate=0.01)
    # reload the model to check the weights
    NNagent = tf.keras.models.load_model(path_save_model)
    print((NNagent.weights))
