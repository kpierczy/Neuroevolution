import os
import gym
import numpy as np
from tensorflow import keras
from DQNAgent import DQNAgent

#------------------------------------------------------------------#
#---------------------------- Configuration -----------------------#
#------------------------------------------------------------------#

# GPU usage config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Training environment
env = gym.make('CartPole-v1')
# Directory for model's saves
modelsDir = 'python/q-learning/models/'


# Number of games to play
gamesNum = 1000
# Maximum games length
gamesLengthMax = 5000
# Penalty for losings
penalty = -10

# Agent's parameters
memSize      = 2000

discount     = 0.95
discountRise = 1
discountMax  = 0.95

epsilon      = 1.0
epsilonDecay = 0.995
epsilonMin   = 0.01

# Learning parameters
batchSize  = 8
iterations = 8

learningRate = 1e-3
optimiser  = keras.optimizers.Adam(learning_rate=learningRate)
loss       = 'mse'

# Model's internal layers
layerStack = (
    keras.layers.Dense(36, activation='relu'),
    keras.layers.Dense(36, activation='relu')
)

# Score threshold for model's save
threshold = 400

# Display training simulations
display = False


#------------------------------------------------------------------#
#--------------------------- Initialization -----------------------#
#------------------------------------------------------------------#

agent = DQNAgent(
    env.observation_space.shape[0], env.action_space.n, memSize=memSize,
    gamma=discount, gammaRise=discountRise, gammaMax=discountMax,
    epsilon=epsilon, epsilonDecay=epsilonDecay, epsilonMin=epsilonMin
)

agent.initialize(layerStack, loss, optimiser)


#------------------------------------------------------------------#
#------------------------------ Training --------------------------#
#------------------------------------------------------------------#

for gameNum in range(gamesNum):

    # Reset environment
    state = env.reset()
    state = np.reshape(state, [1, state.shape[0]])

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
            print("Episode: {}/{}, Score: {}, epsilon: {:.2}".format(gameNum + 1, gamesNum, score - penalty, agent.epsilon))
            break

    # Save model
    if score >= threshold:
        agent.save(modelsDir + 'score_{}'.format(score - penalty))

    # Teach model
    if len(agent.memory) > batchSize:
        agent.learn(batchSize, iterations)
    