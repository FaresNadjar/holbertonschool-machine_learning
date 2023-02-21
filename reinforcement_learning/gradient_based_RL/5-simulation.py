#!/usr/bin/env python3
"""
    NN agent
"""
import numpy as np
import tensorflow as tf

encode_state = __import__('2-encode_state').encode_state

def simulate_one_step(env, step_state, model_path):
    # load the model
    NNagent = tf.keras.models.load_model(model_path)
    # encode state
    step_state_encoded = np.array([encode_state(step_state)])
    # make prediction
    prediction = NNagent.predict(step_state_encoded)
    if step_state == 2:
      action = 0 # do nothing if the stock is full even with a selling event
    else:
      # choose the action index with maximum value
      action = np.argmax(prediction)
    # take a step in the environment and return the output
    next_state, reward, done, info = env.step(action)

    return action, next_state, reward, info
