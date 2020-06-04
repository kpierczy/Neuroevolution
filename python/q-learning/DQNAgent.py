import numpy as np
from collections import deque
from tensorflow import keras
import random

class DQNAgent:
    """
    DQN (Deep Q Network) agent class utilizing Keras programming model.
    Model implement's DeepMind-like RL algorithm sharing simple learning
    and usage interface.
    """




    """
    Constructor. Initializes DQN agent with given parameters.
    
    @note : constructor doesn't create internal neural net model. 
            To do this one has to call initialize() method

    @param statesNum : number of the states agent can observe
    @param actionsNum : number of the actions agent can take
    @param memSize : size of the internal buffor containing
        historical data used during training
    @param gamma : discount rate; the higher the rate is the
        more actions are taken with respect to future reward
        rather than immediate one
    @param gammaRise : parameter controlling discount rate 
        change from one learning session to another
    @param gammaMax : maximum value of the discount rate
    @param epsilon : probability of the performing a random 
        action by the agent rather than one resulting from
        neural net model
    @param epsilonDecay : parameter controlling epsilon change 
        from one learning session to another
    @param epsilonMin : minimum value of the epsilon

    """
    def __init__(self, statesNum, actionsNum, memSize=2000,
                gamma=0.95, gammaRise=1, gammaMax=0.95,
                epsilon=1.0, epsilonDecay=0.995, epsilonMin=0.01):

        # Size of the net's input and output
        self.statesNum = statesNum
        self.actionsNum = actionsNum

        # Double-ended queue that stores <s_t, a_t, r_t, s_{t+1}> touples registered
        # by the agent. Touple are used to feed backprop learning process.
        #   -> s_t     - state in the 't' moment
        #   -> a_t     - action taken in the 't' moment
        #   -> r_t     - reward aquired in the 't' moment
        #   -> s_{t+1} - state in the 't+1' moment
        self.memory = deque(maxlen=memSize) 

        # Discount rate parameters
        self.gamma = gamma 
        self.gammaRise = gammaRise
        self.gammaMax = gammaMax

        # Epsilon parameters - non-zero epsilon makes possibility to take
        # a random action by the agent rather than following net's result 
        self.epsilon = 1.0
        self.epsilonDecay = 0.995
        self.epsilonMin = 0.01

        # Internal model
        self.__model = False
    



    """
    (Re)Initialize internal model with given parameters.

    @param layers : list of the Keras internal layers
    @param loss function : Keras loss function
    @param optimizer : Keras optimizer
    @param reinitialize : if True object will be reinitialized

    @returns : true if model was (re)initialized, false otherwise

    """
    def initialize(self, layers, lossFunction, optimizer, reinitialize=False):
        
        if not (self.__model) or (reinitialize):
            
            # Declare model's input shape
            inputs = keras.Input(shape=(self.statesNum))

            # Stack internal layers
            layerStack = layers[0](inputs)
            for layerNum in range(1, len(layers)):
                layerStack = layers[layerNum](layerStack)

            # Declare model's output
            outputs = keras.layers.Dense(self.actionsNum, activation='linear')(layerStack)

            # Create and compile model
            self.__model = keras.Model(inputs=inputs, outputs=outputs)
            self.__model.compile(loss=lossFunction, optimizer=optimizer)

            return True
            
        else:
            return False




    """
    Saves single iteration data to the internal log

    @param state : state in the iteration
    @param action : action take in the iteration
    @param reward : reward obtain in the iteration
    @param next_state : state observed after taking an action
    @param done : true if iteration was last in the episode

    """
    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))




    """
    Performs action choosen for the given state

    @param state : actual state

    """
    def act(self, state):
    
        # Decide to act randomly or not
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.actionsNum)
        else:

            actValues = self.__model(state.reshape((1, self.statesNum))).numpy()
            return np.argmax(actValues[0])




    """
    Performs a number of learning batches basing of the historical
    data. Records are sampled randomly from the saved records.

    @param batchSize : size of the batch
    @param iterations : number of batches

    """
    def learn(self, batchSize, iterations):

        for _ in range(iterations):

            # Sample records for the batch
            samples = random.sample(self.memory, batchSize)

            # Prepare batch
            batchInput  = np.zeros((batchSize, self.statesNum))
            batchOutput = np.zeros((batchSize, self.actionsNum))
            record = 0
            for state, action, reward, nextState, done in samples:
                
                # Reshape states to the Keras requirements
                state = state.reshape((1, self.statesNum))
                nextState = nextState.reshape((1, self.statesNum))

                # If done, target reward is the historical reward
                targetReward = reward
                # Otherwise compute target as the sum of historical
                # reward and weighted future estimation
                if not done:
                    targetReward = (reward + self.gamma * np.amax(self.__model(nextState).numpy()[0])) 

                target = self.__model(state).numpy()
                target[0][action] = targetReward

                # Save target to the batch
                batchInput[record] = state[0]
                batchOutput[record] = target[0]

            self.__model.fit(batchInput, batchOutput, epochs=1, verbose=0)

        # Update epsilon
        if self.epsilon * self.epsilonDecay >= self.epsilonMin:
            self.epsilon *= self.epsilonDecay

        # Update discount rate
        if self.gamma * self.epsilonDecay <= self.gammaMax:
            self.gamma *= self.gammaRise




    """
    Loads model from the file

    @param name : path to the file to save

    """
    def load(self, name):
        self.__model.load_model(name)

    """
    Saves model to the file

    @param name : path to the file to load from
    
    """
    def save(self, name):
        self.__model.save(name)