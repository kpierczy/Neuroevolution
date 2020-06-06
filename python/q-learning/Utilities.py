# Decorator used to store C-like static variables
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(epsilon=1)
def LinearEpsilon(frameNum, stableTime=50000, initial=1,
                  firstMin=0.1, firstDecTime=950000,
                  SecondMin=0.01, secondDecTime=100000):

    """
    Function manages epsilon value in a linear trend. Firstly,
    epsilon is hold at 'initial' for 'stableTime' frames. Next, it
    linearly decreases to the 'firstMin' within 'firstDecTime'
    frames. Finally it decreases to the 'secondMin' within
    'secondDecTime' frames and at is hold at this value
    for all future calls

    Note:
        All but first arguments should be binded if function
        is to be used with DQNAgent class

    Args:
        frameNum : Integer, index of the actual frame
        initial : float, value of the epsilon for the 'stableTime'
        stableTime : Integer, number of frames that epsilon is
            hold at 1
        firstMin : value of the epsilon after the first linear
            descend
        firstDecTime : number of frames that 'firstMin' values
            is reached within
        secondMin : value of the epsilon after the second linear
            descend
        secondDecTime : number of frames that 'secondMin' values
            is reached within

    Return:
        float, value of the epsilonf or current frame

    """

    if frameNum < stableTime:
        return initial
    elif frameNum < stableTime + firstDecTime:
        return initial - (initial - firstMin) / firstDecTime * (frameNum - stableTime)
    elif frameNum < stableTime + firstDecTime + secondDecTime:
        return firstMin - (firstMin - SecondMin) / secondDecTime * (frameNum - stableTime - firstDecTime)
    else:
        return secondDecTime



