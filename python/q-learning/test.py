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
# Path to the model
modelName = 'python/q-learning/models/score_499.0'


#------------------------------------------------------------------#
#--------------------------- Initialization -----------------------#
#------------------------------------------------------------------#

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
agent.load(modelName)


#------------------------------------------------------------------#
#------------------------------ Training --------------------------#
#------------------------------------------------------------------#

while True:

    # Reset environment
    state = env.reset()
    state = np.reshape(state, [1, state.shape[0]])

    # Play a game
    score = 0
    while True:

        # Render scene
        env.render()

        # Interact with environment
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

        # Update score
        score += 1

        # Print summary if done
        if done:
            print("Score{}".format(score))
            break