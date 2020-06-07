import os
import numpy as np
import tensorflow as tf



def session(env, config, agent):
    
    """
    Generic training session for an arbitrary RL algorithm. session() manages
    saving & loading models as well Tensorboard updating. Every 'agent' object
    that meets some 

    """

    # Create folder for the models
    savesDir = os.path.join(config['paths']['savesDir'], env.unwrapped.spec.id)
    if not os.path.exists(savesDir):
        os.mkdir(savesDir)
    if not os.path.exists(os.path.join(savesDir, 'models')):
        os.mkdir(os.path.join(savesDir, 'models'))
    if not os.path.exists(os.path.join(savesDir, 'replays')):
        os.mkdir(os.path.join(savesDir, 'replays'))

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
    print("Environment: {}                                                  ".format(config['environment']))
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
                    if info['ale.lives'] < lives:
                        liveLost = True
                    else:
                        liveLost = False

                    # Save interaction result to the history set
                    agent.observe(action, reward, state, liveLost)

                    # Teach model
                    if framesNum > config['agent']['initialReplayMemorySize'] and \
                        framesNum % config['time']['learningFrequency'] == 0:
                        history = agent.learn(verbose=config['log']['verboseLearning'])
                        lossLog.append(history.history['loss'])

                    # Break game if done
                    if done:
                        break

                # Save reward
                trainingRewards.append(episodeReward)

                # Update training stats on the Tensorboard
                if framesNum > config['agent']['initialReplayMemorySize'] and \
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

            for _ in range(config['time']['evaluationGames']):

                # Reset environment
                agent.stateReset()
                state = env.reset()
                episodeReward = 0

                for i in range(config['time']['maxFramesPerGame']):

                    # Display render
                    if config['log']['displayEval']:
                        env.render()

                    # Interact with environment
                    action = agent.act(state, evaluation=True, frameKeep=config['agent']['evaluationFrameKeep'])
                    state, _, done, _ = env.step(action)
                    episodeReward += reward

                    # End of the game
                    if done:
                        break

            # Save reward
            evaluationRewards.append(episodeReward)

            # Update training stats on the Tensorboard
            with summaryWriter.as_default():
                tf.summary.scalar('evaluation_score', np.mean(evaluationRewards[-config['time']['evaluationGames']:]), framesNum)

            # Evaluation summary
            avgEvalReward = np.mean(evaluationRewards[-config['time']['evaluationGames']:])
            print("\n")
            print("=================================================")
            print("| Average evaluation score: {}                  |".format(avgEvalReward))
            print("=================================================")
            print("\n")

            # Save model & replay memory
            agent.save(os.path.join(
                config['paths']['savesDir'],
                env.unwrapped.spec.id,
                'models',
                'avg_eval_{}'.format(avgEvalReward)
            ))
            
            # Save replay memory
            replayMemoryDir = os.path.join(
                config['paths']['savesDir'],
                env.unwrapped.spec.id,
                'replays',
                'avg_eval_{}'.format(avgEvalReward)
            )
            if not os.path.exists(replayMemoryDir):
                os.mkdir(replayMemoryDir)
            np.save(os.path.join(replayMemoryDir, 'actions'), agent.replayMemory.actions)
            np.save(os.path.join(replayMemoryDir, 'dones'), agent.replayMemory.dones)
            np.save(os.path.join(replayMemoryDir, 'rewards'), agent.replayMemory.rewards)
            np.save(os.path.join(replayMemoryDir, 'states'), agent.replayMemory.states) 


    #--------------------------------------------------------#
    #   If exception occured, save model and replay memory   #
    #--------------------------------------------------------#

    finally:

        # Save model & replay memory
        agent.save(os.path.join(
            config['paths']['savesDir'],
            env.unwrapped.spec.id,
            'models',
            'interrupted_avg_eval_{}'.format(avgEvalReward)
        ))

        # Save replay memory
        replayMemoryDir = os.path.join(
            config['paths']['savesDir'],
            env.unwrapped.spec.id,
            'replays',
            'interrupted_avg_eval_{}'.format(avgEvalReward)
        )
        if not os.path.exists(replayMemoryDir):
            os.mkdir(replayMemoryDir)
        np.save(os.path.join(replayMemoryDir, 'actions'), agent.replayMemory.actions)
        np.save(os.path.join(replayMemoryDir, 'dones'), agent.replayMemory.dones)
        np.save(os.path.join(replayMemoryDir, 'rewards'), agent.replayMemory.rewards)
        np.save(os.path.join(replayMemoryDir, 'states'), agent.replayMemory.states)        
