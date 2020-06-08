"""
   Filename : BreakoutEnv.cpp
       Date : Mon June 08 2020
     Author : Krzysztof Pierczyk
    Version : 1.0

Description : Simple wrapper around 'Breakout' gym environment
"""

import gym

class BreakoutEnv(object):
    
    """ 
    Wrapper for the Breakout environment. The aim of the wrapper is to count
    lives of the agent and return additional information when life is lost.
    """
    
    def __init__(self, envName):
        self.env = gym.make(envName)
        self.unwrapped = self.env.unwrapped
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.lastLives = 0

    def reset(self):
        self.lastLives = 0
        return self.env.reset()

    def step(self, action):

        state, reward, done, info = self.env.step(action)

        # Update lives info
        if info['ale.lives'] < self.lastLives:
            info['lifeLost'] = True
        else:
            info['lifeLost'] = False
        self.lastLives = info['ale.lives']
            
        return state, reward, done, info

    def render(self):
        self.env.render()