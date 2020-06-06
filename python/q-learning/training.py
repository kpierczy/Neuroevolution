import os
import sys
from pathlib     import Path
from collections import deque

import gym
import numpy    as np
from tensorflow import keras
from DQNAgent import DQNAgent

#------------------------------------------------------------------#
#---------------------------- Configuration -----------------------#
#------------------------------------------------------------------#

# GPU usage config (HIP for AMD; CUDA for Nvidia)(-1 for CPU; 0,1,... for GPUs)
os.environ["HIP_VISIBLE_DEVICES"]  = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Training environment
env = gym.make('CartPole-v1')
# Directory for model's saves
modelsDir = 'python/q-learning/models/'
# Model to load (optional: False -> create new model)
modelPath = False
# Name of the final model save
finalName = 'python/q-learning/models/CartPole-v1/score_367.0'

# Number of games to play
gamesNum = 1000
# Maximum games length
gamesLengthMax = 5000
# Penalty for losings
penalty = -5
# Number of last episodes that an average score is computed from
avgLen = 30
# Score threshold for model's save
threshold = 250

# Policy for epsilon management
epsilonPolicy = lambda frameNum : 0.1*0.9995**frameNum

# Agent's memory
memSize = 10000
# Number of states that are stacked together to make agent's state
stackedStateLength = 1

# Learning data parameters
batchSize  = 32
# Learning features
learningRate = 1e-3
optimizer    = keras.optimizers.Adam(learning_rate=learningRate)
loss         = 'mse'
# Number of frames that action is hold stable
frameSkip = 1

# Model's stack
inputs = keras.Input(shape=np.concatenate((np.array([stackedStateLength]), env.observation_space.shape), axis=0))
layerStack = keras.layers.Flatten()(inputs)
layerStack = keras.layers.Dense(128, activation='relu')(layerStack)
layerStack = keras.layers.Dense(128, activation='relu')(layerStack)
layerStack = keras.layers.Dense(env.action_space.n, activation='linear')(layerStack)

# Display training simulations
display = True


#------------------------------------------------------------------#
#--------------------------- Initialization -----------------------#
#------------------------------------------------------------------#

# Initialize a new model
if not modelPath:
    agent = DQNAgent(
        inputs, layerStack, memSize=memSize, stackedStateLength=stackedStateLength,
        epsilonPolicy=epsilonPolicy, optimizer=optimizer, loss=loss, batchSize=batchSize
    )
# ... or load the old one for futher learning
else:
    agent = DQNAgent(
        env.observation_space.shape, env.action_space.n, layerStack
    )
    agent.load(modelPath)

# Create folder for the models
modelsDir = os.path.join(modelsDir, env.unwrapped.spec.id)
if not os.path.exists(modelsDir):
    print(os.getcwd())
    os.mkdir(modelsDir)

# Initialize average score window
scoreIdx  = 0
scoresWin = np.zeros((avgLen))

#------------------------------------------------------------------#
#------------------------------ Training --------------------------#
#------------------------------------------------------------------#

for gameNum in range(gamesNum):

    # Reset environment
    state = env.reset()

    # Play a game
    score = 0
    frameSkipCounter = 0
    for _ in range(gamesLengthMax):

        # Display render
        if display:
            env.render()

        # Interact with environment
        if frameSkipCounter == 0:
            action = agent.act(np.repeat(state, stackedStateLength))
        frameSkipCounter += 1
        if frameSkipCounter == frameSkip + 1:
            frameSkipCounter = 0
        state, reward, done, _ = env.step(action)

        # Assign penlaty if agent lost
        reward = reward if not done else penalty

        # Save interaction result to the history set
        agent.observe(action, reward, state, done)

        # Update score
        score += reward

        # Teach model
        if agent.replayMemory.count > stackedStateLength:
            agent.learn(verbose=0)

        # Print summary if done
        if done:
            scoresWin[scoreIdx] = score - penalty
            scoreIdx += 1
            scoreIdx %= avgLen
            break

    # Print score
    if gameNum + 1 < avgLen:
        print("Episode: {}/{}, Score: {:6.3f}, Average: {:.3f}, epsilon: {:.2f}".format(
            gameNum + 1,
            gamesNum,
            scoresWin[scoreIdx - 1],
            np.mean(scoresWin[:scoreIdx]),
            agent.epsilonPolicy(agent.observationsSeen)
        ))
    else:
        print("Episode: {}/{}, Score: {}, Average: {:.3f}, epsilon: {:.2f}".format(
            gameNum + 1,
            gamesNum,
            scoresWin[scoreIdx - 1],
            np.mean(scoresWin),
            agent.epsilonPolicy(agent.observationsSeen)
        ))

    # Save model
    if score >= threshold:
        agent.save(os.path.join(modelsDir, 'score_{}'.format(score - penalty)))

# Save final model 
agent.save(modelsDir + finalName)
    