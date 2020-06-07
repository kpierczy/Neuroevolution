import sys
sys.path.insert(0, 'python')
print(sys.path)
import os
import gym
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from DQNAgent import DQNAgent
from utilities import linearEpsilon
from common.session import session


#===================================================================================#
#================================== Configuration ==================================#
#===================================================================================#

# Load configuration
with open('python/q-learning/trainingParams.json') as configFile:
    config = json.load(configFile)

#===================================================================================#
#================================== Initialization =================================#
#===================================================================================#

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
            initialPeriod=config['agent']['epsilonPolicyConfig']['linear']['initialPeriod'],
            firstTarget=config['agent']['epsilonPolicyConfig']['linear']['firstTarget'],
            firstTargetPeriod=config['agent']['epsilonPolicyConfig']['linear']['firstTargetPeriod'],
            finalTarget=config['agent']['epsilonPolicyConfig']['linear']['finalTarget'],
            finalTargetPeriod=config['agent']['epsilonPolicyConfig']['linear']['finalTargetPeriod'],
        )
elif config['agent']['epsilonPolicy'] == 'exponential':
    epsilonPolicy = lambda frameNum : np.multiply(
        config['agent']['epsilonPolicyConfig']['exponential']['initial'],
        config['agent']['epsilonPolicyConfig']['exponential']['decay']**frameNum
    )

# Optimiser
optimizer = keras.optimizers.Adam(learning_rate=config['model']['learningRate'])
if config['model']['optimizer'] == 'SGD':
    optimizer = keras.optimizers.SGD(learning_rate=config['model']['learningRate'])
elif config['model']['optimizer'] == 'RMSprop':
    optimizer = keras.optimizers.RMSprop(learning_rate=config['model']['learningRate'])
elif config['model']['optimizer'] == 'Adadelta':
    optimizer = keras.optimizers.Adadelta(learning_rate=config['model']['learningRate'])
elif config['model']['optimizer'] == 'Adamax':
    optimizer = keras.optimizers.Adamax(learning_rate=config['model']['learningRate'])
elif config['model']['optimizer'] == 'Adagrad':
    optimizer = keras.optimizers.Adagrad(learning_rate=config['model']['learningRate'])

# Model's input
inputs = keras.Input(shape=np.concatenate((np.array([config['agent']['stackedStateLength']]), env.observation_space.shape), axis=0))
layerStack = inputs
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
        )(layerStack)

    # Flatten layer
    elif layer['type'] == 'flatten':
        layerStack = keras.layers.Flatten()(layerStack)
        
# Model's output
layerStack = keras.layers.Dense(
    env.action_space.n, activation='linear',
    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
)(layerStack)


# Initialize a new model
if not config['paths']['initialModelName']:
    agent = DQNAgent(
        inputs, layerStack, memSize=config['agent']['replayMemorySize'],
        stackedStateLength=config['agent']['stackedStateLength'],
        epsilonPolicy=epsilonPolicy, optimizer=optimizer,
        loss=config['model']['lossFunction'], batchSize=config['model']['batchSize'],
        modelName=config['model']['name']
    )
# Or load the old one for futher learning
else:
    agent = DQNAgent(
        env.observation_space.shape, env.action_space.n, layerStack
    )
    agent.load(os.path.join(
        config['paths']['savesDir'],
        env.unwrapped.spec.id,
         'models',
         config['paths']['initialModelName']
    ))


# Load replay memory if needed
if config['paths']['initialReplayMemoryName'] != False:
    replayMemoryPath = os.path.join(
        config['paths']['savesDir'],
        env.unwrapped.spec.id,
        'replays',
        config['paths']['initialReplayMemoryName']
    )
    agent.replayMemory.states = np.load(os.path.join(replayMemoryPath, 'actions'))
    agent.replayMemory.states = np.load(os.path.join(replayMemoryPath, 'dones'))
    agent.replayMemory.states = np.load(os.path.join(replayMemoryPath, 'rewards'))
    agent.replayMemory.states = np.load(os.path.join(replayMemoryPath, 'states'))


#===================================================================================#
#====================================== Training ===================================#
#===================================================================================#

session(env, config, agent)