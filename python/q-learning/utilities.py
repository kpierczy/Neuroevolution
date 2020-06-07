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
        return finalTargetPeriod
