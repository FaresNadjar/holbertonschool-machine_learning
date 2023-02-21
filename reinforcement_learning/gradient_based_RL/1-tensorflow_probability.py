#!/usr/bin/env python3
"""
    NN agent
"""
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


def get_action(model, state):
    """
        Get action
    """
    prob = model(np.array([state]))
    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    # sample for 1 step
    action_sample = dist.sample()
    return dist, int(action_sample)
