#!/usr/bin/env python3
"""
    NN agent
"""
import tensorflow as tf


def build_NNagent():
    """
        Build the NN agent
    """
    initializer = tf.keras.initializers.HeNormal()
    inputs = tf.keras.Input(shape=(3,))
    hiddenLayer = tf.keras.layers.Dense(5, activation='relu', kernel_initializer=initializer)(inputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(hiddenLayer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
