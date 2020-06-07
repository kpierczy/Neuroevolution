"""
   Filename : utilities.cpp
       Date : Sun June 07 2020
     Author : Krzysztof Pierczyk
    Version : 1.0

Description : Set of the utilities used during DQNAgent training
"""

import os

def static_vars(**kwargs):

    """ Decorator used to store C-like static variables """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

    

def linearEpsilon(frameNum,  initial=1, initialPeriod=50000,
                  firstTarget=0.1, firstTargetPeriod=950000,
                  finalTarget=0.01, finalTargetPeriod=100000):

    """
    Function manages epsilon value in a linear trend. Firstly,
    epsilon is hold at 'initial' for 'stableTime' frames. Next, it
    linearly decreases to the 'firstTarget' within 'firstTargetPeriod'
    frames. Finally it decreases to the 'finalTarget' within
    'finalTargetPeriod' frames and at is hold at this value
    for all future calls

    Note:
        All but first arguments should be binded if function
        is to be used with DQNAgent class

    Args:
        frameNum : Integer, index of the actual frame
        initial : float, value of the epsilon for the 'stableTime'
        initialPeriod : Integer, number of frames that epsilon is
            hold at 1
        firstTarget : value of the epsilon after the first linear
            descend
        firstTargetPeriod : number of frames that 'firstTarget' values
            is reached within
        finalTarget : value of the epsilon after the second linear
            descend
        finalTargetPeriod : number of frames that 'finalTarget' values
            is reached within

    Return:
        float, value of the epsilonf or current frame

    """

    if frameNum < initialPeriod:
        return initial
    elif frameNum < initialPeriod + firstTargetPeriod:
        return initial - (initial - firstTarget) / firstTargetPeriod * (frameNum - initialPeriod)
    elif frameNum < initialPeriod + firstTargetPeriod + finalTargetPeriod:
        return firstTarget - (firstTarget - finalTarget) / finalTargetPeriod * (frameNum - initialPeriod - firstTargetPeriod)
    else:
        return finalTarget


def save(name, parentFolder, agent):
    
    """
    Saves agent's model and replay memory to the file established by the configuration
    file and the name

    Args:
        parentFolder : path to the folder with models and replays
        agent : DQNAgent, agent to save
        name : name for the performed save
    """

    # Check if directories for models and replays exist. If don't, create them
    modelsPath = os.path.join(parentFolder, 'models')
    replaysPath = os.path.join(parentFolder, 'replays')
    if not os.path.exists(modelsPath):
        os.mkdir(modelsPath)
    if not os.path.exists(replaysPath):
        os.mkdir(replaysPath)

    # Save model
    agent.saveModel(os.path.join(modelsPath, name))

    # Save memory
    agent.saveMemory(os.path.join(replaysPath, name))

