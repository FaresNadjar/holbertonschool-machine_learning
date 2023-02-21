#!/usr/bin/env python3

"""
    Main file
"""

if __name__ == '__main__':
    import numpy as np

    build_NNagent = __import__('0-build_NN_agent').build_NNagent
    
    model = build_NNagent()
    model.summary()

    print("Let's check the output of our model")
    state = [1, 0, 0]
    model_output = model(np.array([state]))
    print(model_output)