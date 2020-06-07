import os
import gym
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from DQNAgent import DQNAgent
from utilities import linearEpsilon

#------------------------------------------------------------------#
#---------------------------- Configuration -----------------------#
#------------------------------------------------------------------#

# Load configuration
config = json.load('python/q-learning/trainingParams.json')


#------------------------------------------------------------------#
#--------------------------- Initialization -----------------------#
#------------------------------------------------------------------#

# GPU usage config (HIP for AMD; CUDA for Nvidia)(-1 for CPU; 0,1,... for GPUs)
os.environ["HIP_VISIBLE_DEVICES"]  = config['computeDevices']
os.environ["CUDA_VISIBLE_DEVICES"] = config['computeDevices']


# Create training environment
env = gym.make(config['environment'])


# Policy for epsilon management
epsilon = lambda frameNum : 0.5
if config['agent']['epsilonPolicy'] == 'linear':
    epsilonPolicy = lambda frameNum : linearEpsilon(frameNum, 
            initial=config['agent']['epsilonPolicyConfig']['linear']['initial'],
            initialPeriod=config['initialPeriod']['agent']['linear']['epsilonPolicyConfig'],
            firstTarget=config['firstTarget']['agent']['linear']['epsilonPolicyConfig'],
            firstTargetPeriod=config['firstTargetPeriod']['agent']['linear']['epsilonPolicyConfig'],
            finalTarget=config['finalTarget']['agent']['linear']['epsilonPolicyConfig'],
            finalTargetPeriod=config['finalTargetPeriod']['agent']['linear']['epsilonPolicyConfig'],
        )
elif config['agent']['epsilonPolicy'] == 'exponential':
    epsilonPolicy = lambda frameNum : np.multiply(
        config['agent']['epsilonPolicyConfig']['exponential']['initial'],
        config['agent']['epsilonPolicyConfig']['exponential']['decay']**frameNum
    )


# Optimiser
optimiser = keras.optimizers.Adam(learning_rate=config['model']['learningRate'])
if config['model']['optimiser'] == 'SGD':
    optimiser = keras.optimizers.SGD(learning_rate=config['model']['learningRate'])
elif config['model']['optimiser'] == 'RMSprop':
    optimiser = keras.optimizers.RMSprop(learning_rate=config['model']['learningRate'])
elif config['model']['optimiser'] == 'Adadelta':
    optimiser = keras.optimizers.Adadelta(learning_rate=config['model']['learningRate'])
elif config['model']['optimiser'] == 'Adamax':
    optimiser = keras.optimizers.Adamax(learning_rate=config['model']['learningRate'])
elif config['model']['optimiser'] == 'Adagrad':
    optimiser = keras.optimizers.Adagrad(learning_rate=config['model']['learningRate'])


# Model's input
layerStack = keras.Input(shape=np.concatenate((np.array([config['agent']['stackedStateLength']]), env.observation_space.shape), axis=0))
# Model's stack
for layer in config['model']['layers']:
    # Dense layer
    if layer['type'] == 'dense':

        # Define initializer (one available at now)
        if layer['initializer'] == 'varianceScaling':
            initializer = keras.initializers.VarianceScaling(
                scale=layer['scale']
            )
        # Stack layer
        layerStack = keras.layers.Dense(
            layer['units'], activation=layer['activation'],
            kernel_initializer=layer['activation'],
            initializer=initializer
        )

    # Flatten layer
    elif layer['type'] == 'flatten':
        layerStack = keras.layers.Flatten()
        
# Model's output
layerStack = keras.layers.Dense(
    env.action_space.n, activation='linear',
    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
)(layerStack)


# Initialize a new model
if not config['paths']['initialModelName']:
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
    lives = 0
    for _ in range(gamesLengthMax):

        # Display render
        if display:
            env.render()

        # Interact with environment
        action = agent.act(state, frameKeep=frameKeep)
        state, reward, done, info = env.step(action)

        # Make agent treat each lost life as the episode's end
        if info['ale.lives'] < lives:
            liveLost = True
        else:
            liveLost = False

        # Assign penlaty if agent a lostlife
        reward = reward if not done else liveLost

        # Save interaction result to the history set
        agent.observe(action, reward, state, liveLost)

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
    