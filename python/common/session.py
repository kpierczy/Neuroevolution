import os
import random
import numpy as np
import tensorflow as tf

"""
   Filename : session.cpp
       Date : Mon June 08 2020
     Author : Krzysztof Pierczyk
    Version : 1.1

Description : Simple framework desired to train an arbitrary RL algorithm with
              saves automatisation and Tensorboard management
"""

def session(env, config, agent, save):
    
    """
    Generic training session for an arbitrary RL algorithm. Function implements
    a training loop that contains training epochs and evaluation epochs. After
    every evaluation a model and replay memory are saved to the files.
    
    session() manages saving models with the given policy as well Tensorboard 
    updating. Every 'agent' object that meets some interface requirements can 
    use the the session() training.

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
            * epsilonPolicy : 
                handler to the function 'fun(frameNum)' returning epsilon value
                for the given frames number
            * observationsSeen :
                foregoing number of observe() calls
                
            - stateReset() : 
                resets agent's state between games
            - act(state, frameKeep=1, evaluation=False) : 
                returns agent's action for the given state, Function takes two
                keyword arguments. 'frameKeep' states for how many iterations
                action is hold without a change. 'evaluation' states if
                act(...) is called in the avaluation or training context
            - observe(action, reward, state, done) : 
                saves observation to the agent's memory. 'action' is the action
                taken in the current iteration, 'reward' is the reward acquired
                after taking the 'action', 'state' is the state observed AFTER
                performing the 'action' and 'done' is a boolean which is true
                if the game iterations terminated
            - learn(**kwargs) : 
                keyword arguments passed to the keras.Model.fit(...) function
        save : function handler with save(name) header saving the model given
            a name


    """

    # initialize evaluation score
    avgEvalReward = 0

    # Tensorboard writer
    summaryWriter = tf.summary.create_file_writer(
        os.path.join(config['log']['directory'], config['log']['logID'])
    )


    #===================================================================================#
    #========================================= Info ====================================#
    #===================================================================================#

    print('\n\n')
    print("=================================================================")
    print("|                         Training session                      |")
    print("=================================================================")
    print("Environment: {}                                                  ".format(config['env']))
    print("                                                                 ")
    print("Available actions: {}                                            ".format(env.unwrapped.get_action_meanings()))
    agent.model.summary()
    print("\n\n\n")


    #===================================================================================#
    #====================================== Training ===================================#
    #===================================================================================#

    try:

        # Initialize statistics containers
        lossLog = []
        trainingRewards = []

        framesNum = 0
        while framesNum < config['time']['maxFramesNum']:

            #----------------------------------------------------------------#
            #                          Training session                      #
            #----------------------------------------------------------------#

            trainingSessionFrames = 0

            while trainingSessionFrames < config['time']['evaluationFreq']:

                # Reset game
                agent.stateReset()
                state = env.reset()
                episodeReward = 0
                lives = 0

                for _ in range(config['time']['maxFramesPerGame']):

                    # Increment frames counters
                    framesNum += 1
                    trainingSessionFrames +=1

                    # Display render
                    if config['log']['display']:
                        env.render()

                    # Interact with environment
                    action = agent.act(state, frameKeep=config['agent']['trainingFrameKeep'])
                    state, reward, done, info = env.step(action)
                    episodeReward += reward

                    # Clip reward
                    if config['agent']['clipReward']:
                        if reward > 0:
                            reward = 1
                        elif reward < 0:
                            reward = -1

                    # Make agent treat each lost life as the episode's end
                    if config['environment']['infoAsDone'] != False:
                        if info[config['environment']['infoAsDone']]:
                            episodeTerminated = True
                        else: 
                            episodeTerminated = False
                    else:
                        episodeTerminated = done

                    # Save interaction result to the history set
                    agent.observe(action, reward, state, episodeTerminated)

                    # Teach model
                    if framesNum > config['agent']['initialRandomFrames'] and \
                        framesNum % config['time']['learningFrequency'] == 0:
                        history = agent.learn(verbose=config['log']['verboseLearning'])
                        lossLog.append(history.history['loss'])

                    # Break game if done
                    if done:
                        break

                # Save reward
                trainingRewards.append(episodeReward)

                # Update training stats on the Tensorboard
                if framesNum > config['agent']['initialRandomFrames'] and \
                    len(trainingRewards) % config['log']['trainingLogUpdateFreq'] == 0:

                    with summaryWriter.as_default():
                        tf.summary.scalar('loss', np.mean(lossLog), step=framesNum)
                        tf.summary.scalar('reward', np.mean(trainingRewards[-config['log']['meanTrainingScoreLength']:]), step=framesNum)
                    lossLog = []

                # Print training statistics
                print("Frames: {}/{}, Score: {}, Average: {:.3f}, epsilon: {:.2f}".format(
                    framesNum,
                    config['time']['maxFramesNum'],
                    episodeReward,
                    np.mean(trainingRewards[-config['log']['meanTrainingScoreLength']:]),
                    agent.epsilonPolicy(agent.observationsSeen)
                ))

            #----------------------------------------------------------------#
            #                        Evaluation session                      #
            #----------------------------------------------------------------#

            # Initialize statistics containers
            evaluationRewards = []

            gamesLeft = config['time']['evaluationGames']
            while gamesLeft > 0:

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
                        env.render()

                    # Interact with environment
                    action = agent.act(state, evaluation=True, frameKeep=config['agent']['evaluationFrameKeep'])
                    state, reward, done, info = env.step(action)
                    episodeReward += reward

                    # End of the game
                    if done:
                        gamesLeft -= 1
                        break

                    # Modified terminal state
                    if config['environment']['infoAsDone'] != False:
                        if info[config['environment']['infoAsDone']]:
                            for _ in range(random.randrange(1, config['environment']['evaluationRandomStart'])):
                                if config['log']['displayEval']:
                                    env.render()
                                state, _, _, _ = env.step(config['environment']['evaluationRandomStartAction'])
                            continue

                # Save reward
                evaluationRewards.append(episodeReward)

            # Update training stats on the Tensorboard
            with summaryWriter.as_default():
                tf.summary.scalar('evaluation_score', np.mean(evaluationRewards), framesNum)

            # Evaluation summary
            avgEvalReward = np.mean(evaluationRewards)
            print("\n")
            print("=================================================")
            print("| Average evaluation score: {}                  |".format(avgEvalReward))
            print("=================================================")
            print("\n")

            # Save model
            save('avg_eval_{}'.format(avgEvalReward))


    #--------------------------------------------------------#
    #           If exception occured, save model             #
    #--------------------------------------------------------#

    finally:

        # Save model
        save('interrupted_avg_eval_{}'.format(avgEvalReward))

