#!/usr/bin/env python3
"""
    NN agent
"""
# import the function to build a model
build_NNagent = __import__('0-build_NN_agent').build_NNagent
# import encode_state
encode_state = __import__('2-encode_state').encode_state
# import get_action
get_action = __import__('1-tensorflow_probability').get_action
# import trainAgent
trainAgent = __import__('3-train_NN_agent').trainAgent


def REINFORCE(env, path_save_model, episodes=100, steps=50):
    agentModel = build_NNagent()
    total_reward_episodes = []

    for e in range(episodes):
        done = False
        state = env.reset()
        total_reward_steps = 0
        rewards = []
        states = []
        actions = []
        # encode state
        state = encode_state(state)
        for s in range(steps):
            _, action = get_action(agentModel, state)
            next_state, reward, done, _ = env.step(action)
            # encode next state
            next_state = encode_state(next_state)

            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            total_reward_steps += reward
        total_reward_episodes.append(total_reward_steps)
        trainAgent(states, rewards, actions, path_save_model)
    return total_reward_episodes
