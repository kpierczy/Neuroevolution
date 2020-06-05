import os
import sys
from pathlib     import Path
from collections import deque

import gym
import numpy    as np
from tensorflow import keras
from DQNAgent   import DQNAgent

#------------------------------------------------------------------#
#---------------------------- Configuration -----------------------#
#------------------------------------------------------------------#

# GPU usage config (HIP for AMD; CUDA for Nvidia)(-1 for CPU; 0,1,... for GPUs)
os.environ["HIP_VISIBLE_DEVICES"]  = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Training environment
env = gym.make('Breakout-ram-v0')
# Directory for model's saves
modelsDir = 'python/q-learning/models/'
# Model to load (optional: False -> create new model)
modelPath = 'python/q-learning/models/final'
# Name of the final model save
finalName = 'final'


# Observation space wrapper (optional: False -> no wrapper)
observationWrapper = False
# Shape of the observation space
observationShape = np.array(env.observation_space.shape)

# Agent's memory
memSize      = 10000
# Discount values
discount     = 0.95
discountRise = 1
discountMax  = 0.95
# Epsilon values
epsilon      = 0.9
epsilonDecay = 0.995
epsilonMin   = 0.005

# Learning mode (@see DQNAgent.learn())
mode = 'random'
# Learning data parameters
batchSize  = 10
iterations = 3
epochs     = 1
# Learning features
learningRate = 1e-3
optimiser    = keras.optimizers.Adam(learning_rate=learningRate)
loss         = 'mse'

# Model's internal layers
layerStack = (
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense( 64, activation='relu')
)

# Number of games to play
gamesNum = 1000
# Maximum games length
gamesLengthMax = 5000
# Penalty for losings
penalty = -10
# Number of last episodes that an average score is computed from
avgLen = 100
# Score threshold for model's save
threshold = 20

# Display training simulations
display = False


#------------------------------------------------------------------#
#--------------------------- Initialization -----------------------#
#------------------------------------------------------------------#

# Initialize a new model
if not modelPath:
    agent = DQNAgent(
        observationShape, env.action_space.n, memSize=memSize,
        gamma=discount, gammaRise=discountRise, gammaMax=discountMax,
        epsilon=epsilon, epsilonDecay=epsilonDecay, epsilonMin=epsilonMin
    )
    agent.createModel(layerStack, loss, optimiser)
# ... or load the old one for futher learning
else:
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
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
    for _ in range(gamesLengthMax):

        # Display render
        if display:
            env.render()

        # Interact with environment
        action = agent.act(state)
        nextState, reward, done, _ = env.step(action)

        # Assign penlaty if agent lost
        reward = reward if not done else penalty

        # Save interaction result to the history set
        agent.store(state, action, reward, nextState, done)

        # Pass state
        state = nextState

        # Update score
        score += reward

        # Print summary if done
        if done:
            scoresWin[scoreIdx] = score - penalty
            scoreIdx += 1
            scoreIdx %= avgLen
            break

    # Print score
    if gameNum + 1 < avgLen:
        print("Episode: {}/{}, Score: {}, Average: {:.3f}, epsilon: {:.2f}".format(
            gameNum + 1,
            gamesNum,
            scoresWin[scoreIdx - 1],
            np.mean(scoresWin[:scoreIdx]),
            agent.epsilon
        ))
    else:
        print("Episode: {}/{}, Score: {}, Average: {:.3f}, epsilon: {:.2f}".format(
            gameNum + 1,
            gamesNum,
            scoresWin[scoreIdx - 1],
            np.mean(scoresWin),
            agent.epsilon
        ))

    # Save model
    if score >= threshold:
        agent.save(modelsDir + 'score_{}'.format(score - penalty))

    # Teach model
    if len(agent.memory) >= batchSize * iterations:
        agent.learn(mode=mode, batchSize=batchSize, iterations=iterations, epochs=epochs)

# Save final model 
agent.save(modelsDir + finalName)
    