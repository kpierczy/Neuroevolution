import sys
sys.path.insert(0, 'python')

import os
import gym
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from DQNAgent import DQNAgent
from utilities import linearEpsilon, save
from common.session import session
from common.evaluation import evaluation
from BreakoutEnv import BreakoutEnv


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
env = BreakoutEnv(config['env'])

# Create folders for the models and memory stamps
savesDir = os.path.join(config['paths']['savesDir'], env.unwrapped.spec.id)
if not os.path.exists(savesDir):
    os.mkdir(savesDir)
if not os.path.exists(os.path.join(savesDir, 'models')):
    os.mkdir(os.path.join(savesDir, 'models'))
if not os.path.exists(os.path.join(savesDir, 'replays')):
    os.mkdir(os.path.join(savesDir, 'replays'))

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
            kernel_initializer=initializer,
            bias_initializer=initializer
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
agent = DQNAgent(
    inputs, layerStack, memSize=config['agent']['replayMemorySize'],
    stackedStateLength=config['agent']['stackedStateLength'],
    epsilonPolicy=epsilonPolicy, optimizer=optimizer,
    loss=config['model']['lossFunction'], batchSize=config['model']['batchSize'],
    modelName=config['model']['name']
)
# If required load the old model for futher learning
if config['paths']['initialModelName'] != False:
    modelToLoad = os.path.join(
        config['paths']['savesDir'],
        env.unwrapped.spec.id,
         'models',
         config['paths']['initialModelName']
    )
    agent.loadModel(modelToLoad)

# Load replay memory if needed
if config['paths']['initialReplayMemoryName'] != False:
    replaysToLoad = os.path.join(
        config['paths']['savesDir'],
        env.unwrapped.spec.id,
        'replays',
        config['paths']['initialReplayMemoryName']
    )
    agent.loadMemory(replaysToLoad)

#===================================================================================#
#====================================== Training ===================================#
#===================================================================================#

if config['mode'] == 'training':
    session(env, config, agent, lambda name : save(name, savesDir, agent))
elif config['mode'] == 'evaluation':
    evaluation(env, config, agent)