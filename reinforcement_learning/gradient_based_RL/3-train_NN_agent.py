#!/usr/bin/env python3
"""
    NN agent
"""
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
build_NNagent = __import__('0-build_NN_agent').build_NNagent

def trainAgent(states, rewards, actions, path_save_model, gamma=0.9, l_rate=0.01):
    """
      Train the NN agent
    """
    model = build_NNagent()
    sum_reward = 0
    discnt_rewards = []
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()  

    for state, reward, action in zip(states, discnt_rewards, actions):
        with tf.GradientTape() as tape:
            prob = model(np.array([state]), training=True)
            distribution = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
            log_prob = distribution.log_prob(action)
            loss = -log_prob*reward
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    model.save(path_save_model)
