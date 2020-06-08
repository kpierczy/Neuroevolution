import os
import time
import random
import numpy as np
import tensorflow as tf

"""
   Filename : session.cpp
       Date : Mon June 08 2020
     Author : Krzysztof Pierczyk
    Version : 1.1

Description : Simple framework desired to evaluate trained agent
"""

def evaluation(env, config, agent):

    """
    Simple function used to verify agent's abilities

    Function was tested on the set of Atari environments set, but will probably
    works with other environments.

    Args:
        env : gym environment, an arbitrary gym environment
        config : dictionary, configuration string loaded from the configuration
            file. To acquire informations about required configuration fields
            @see README.md
        agent : object, an arbitrary RL agent sharing some required interface
            ('*' refer to object's member field, when '-' to member methods):

            * model :
                keras model
            - stateReset() : 
                resets agent's state between games
            - act(state, frameKeep=1, evaluation=False) : 
                returns agent's action for the given state, Function takes two
                keyword arguments. 'frameKeep' states for how many iterations
                action is hold without a change. 'evaluation' states if
                act(...) is called in the avaluation or training context

    """

    print('\n\n')
    print("=================================================================")
    print("|                       Evaluation session                      |")
    print("=================================================================")
    print("Environment: {}                                                  ".format(config['env']))
    print("                                                                 ")
    print("Available actions: {}                                            ".format(env.unwrapped.get_action_meanings()))
    agent.model.summary()
    print("\n\n\n")

    while True:

        # Reset environment
        agent.stateReset()
        state = env.reset()
        episodeReward = 0

        # Perform initial random actions
        for _ in range(random.randrange(1, config['environment']['evaluationRandomStart'])):
            if config['log']['displayEval']:
                env.render()
            state, _, _, _ = env.step(config['environment']['evaluationRandomStartAction'])

        for _ in range(config['time']['maxFramesPerGame']):

            # Display render
            if config['log']['displayEval']:
                if config['environment']['envaluationSlowDown'] != False:
                    time.sleep(1 / config['environment']['envaluationSlowDown'])
                env.render()

            # Interact with environment
            action = agent.act(state, evaluation=True, frameKeep=config['agent']['evaluationFrameKeep'])
            state, reward, done, info = env.step(action)
            episodeReward += reward

            # End of the game
            if done:
                break

            # Modified terminal state
            if config['environment']['infoAsDone'] != False:
                if info[config['environment']['infoAsDone']]:
                    for _ in range(random.randrange(1, config['environment']['evaluationRandomStart'])):
                        if config['log']['displayEval']:
                            if config['environment']['envaluationSlowDown'] != False:
                                time.sleep(1 / config['environment']['envaluationSlowDown'])
                            env.render()
                        state, _, _, _ = env.step(config['environment']['evaluationRandomStartAction'])
                    continue