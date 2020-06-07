"""
   Filename : DQNAgent.cpp
       Date : Sat June 06 2020
     Author : Krzysztof Pierczyk
    Version : 1.0

Description : DQN agent implementation based on the DeepMind-like DQN from:
              https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

class DQNAgent:

    """
    DQN (Deep Q Network) agent class utilizing Keras programming model.
    Model implement's DeepMind-like RL algorithm sharing simple learning
    and usage interface.
    """

    class ReplayMemory:

        """ Utility class used to manage replay memory """

        def __init__(self, stateShape, stackedStateLength=4, size=10000,
                     batchSize=32, stateDtype=np.float32):

            """
            Initialize replay memory of the given size

            Args:
                stateShape : np.array, shape of the observed states
                stackedStateLength : number of states stacked together
                    to create an agent's state
                size : Integer, size of the memory buffer
                batchSize : Integer, size of the single training batch
                stateDtype : np.datatype, type of the data representing 
                    states

            """

            self.size = size
            self.stateShape = stateShape
            self.stackedStateLength = stackedStateLength
            self.batchSize = batchSize

            # Create memory buffers
            self.count = 0
            self.current = 0
            self.states  = np.empty(
                np.concatenate((np.array([self.size]), stateShape), axis=0), 
                dtype=stateDtype
            )
            self.actions = np.empty(self.size, dtype=np.int32)
            self.rewards = np.empty(self.size, dtype=np.float32)
            self.dones   = np.empty(self.size, dtype=np.bool)

            # Pre-allocate memory for the mini-batches
            self.trainingStates = np.empty(
                np.concatenate((np.array([self.batchSize, self.stackedStateLength]), stateShape), axis=0),
                dtype=stateDtype
            )
            self.trainingNextStates = np.empty(
                np.concatenate((np.array([self.batchSize, self.stackedStateLength]), stateShape), axis=0),
                dtype=stateDtype
            )
            self.indices = np.empty(self.batchSize, dtype=np.int32)


        def store(self, action, reward, nextState, done):

            """
            Stores a new transition

            Args:
                action : Integer, index of the performed action
                reward : float, obtained reward
                nextState : np.array, state after performing the action
                done : Bolean, parameter stating whether the episode terminated
            """

            if nextState.shape != self.stateShape:
                raise ValueError("State's dimension invalid")

            self.actions[self.current] = action
            self.states[self.current] = nextState
            self.rewards[self.current] = reward
            self.dones[self.current] = done

            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.size


        def _getState(self, idx):

            """ Returns agent's (stacked) state with the given index """

            if self.count == 0:
                ValueError("The replay memory is empty!")
            if idx < self.stackedStateLength - 1:
                raise ValueError("Index must be greater than the size of the state stack!")
            return self.states[idx - self.stackedStateLength + 1 : idx + 1, ...]


        def _getValidIndices(self):

            """
            Writes indices for the next batch to the self.indices.
            Indices are pick randomly and invalid indices are rejected.
            Index can be invalid when:
                (1) state at the som of the 'self.stackedStateLength - 1'
                    states is terminal. Then, stacked state will contain
                    observed states from different games
                (2) index is smaller than 'self.stackedStateLength'
                (3) index is greater than 'self.current' but (index
                    - 'self.stackedStateLength') is smaller than the
                    'self.current'. Then, also stacked state will contain
                    observed states from different games
            """

            for i in range(self.batchSize):
                while True:

                    # Pick random index (index of the 'nextState', not 'state')
                    index = random.randint(self.stackedStateLength, self.count - 1)

                    # Index pointing to the stackedState has to contain observed states from the single game
                    if (index >= self.current) and (index - self.stackedStateLength < self.current):
                        continue
                    if self.dones[index - self.stackedStateLength : index].any():
                        continue
                    break

                # Save index to the pre-allocated array
                self.indices[i] = index


        def getMiniBatch(self):

            """
            Returns mini-batch randomly selected from the memory pool

            Returns: 
                five-elements touple containing:
                    - np.array of shape (self.batchSize, self.stackedStateLength, *(self.stateShape))
                      containing states
                    - np.array of actions
                    - np.array of rewards
                    - np.array of shape (self.batchSize, self.stackedStateLength, *(self.stateShape))
                      containing next states
                    - np.array of dones

            """

            # Check if memory has enugh data to return mini-batch
            if self.count <= self.stackedStateLength:
                raise ValueError('Not enough transitions to get a minibatch')
        
            # Write mini-batch indices to pre-allocated array
            self._getValidIndices()
                
            # Fill pre-loaded mini-batch containers with data
            for idx, stateIdx in enumerate(self.indices):
                self.trainingStates[idx] = self._getState(stateIdx - 1)
                self.trainingNextStates[idx] = self._getState(stateIdx)
            
            return self.trainingStates, self.actions[self.indices], self.rewards[self.indices], self.trainingNextStates, self.dones[self.indices]




    class ImagePreprocesor():

        """ Resizes and converts RGB (if RGB) images to grayscale """
        
        def __init__(self, imageShape=(84, 84), crop=((0, 34), (160, 160))):
            
            """
            Args:
                imageShape : tuple of Integers, size of the preprocessed image in shape
                    (Width, Heigth)
                crop : tuple of two tupples of two Integers, parameters of the cropping
                    process in shape ((xoffset, yoffset), (width, heigth))
            """

            self.imageShape = imageShape
            self.crop = crop

        def __call__(self, image):

            """
            Args:
                image: 2-D or 3-D np.array, image to be preprocessed
            Returns:
                A processed (84, 84, 1) frame in grayscale
            """

            # Check dimensionality
            if len(image.shape) not in (2, 3):
                raise ValueError('Image has to be 2-D or 3-D object!')
            # Convert from RGB to grayscale
            if len(image.shape) == 3:
                image = tf.image.rgb_to_grayscale(image)

            # Perform cropping
            image = tf.image.crop_to_bounding_box(image, self.crop[0][1], self.crop[0][0], self.crop[1][1], self.crop[1][0])

            # Resize image
            return tf.image.resize(image, self.imageShape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)




    def __init__(self, inputs, layerStack, stackedStateLength=4,
                 gammaPolicy=lambda frameNum : 0.95,
                 epsilonPolicy=lambda frameNum : 0.995**frameNum,
                 optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                 loss='mse', batchSize=32, memSize=10000,
                 stateDtype=np.float32, modelName="model", **kwargs):

        """
        Constructor. Initializes DQN agent with given parameters and structure

        Args:
            inputs : tensorflow.keras.Input, model's input
            layerStack : tensorflow.python.framework.ops.Tensor, stack of the
            input and layers constitutig the model
            stackedStateLength : Integer, number of subsequent observed states
                that is combined to obtain agent's state
            gammaPolicy : function handle, discount rate policy, function that
                takes one Integer argument (frame's number) and returns appropriate
                discount rate
            epsilonPolicy : function handle , discount rate policy, function that
                takes one Integer argument (frame's number) and returns appropriate
                discount rate
            optimizer : keras.Optimizer, desired optimiser
            loss : string or keras.losses object, Keras loss function
            batchSize : Integer, size of the single learning mini-batch
            memSize : size of the internal buffer containing historical observations
                used during training (DeepMin nomenclature: replayMemory)
            stateDtype : np.dtype, type of the data that observed states are
                represented with

        Kwargs:
            imageShape : tuple of Integers (optional, default : (84, 84)), size
                of the preprocessed image in shape (Width, Heigth)
            crop : tuple of two tupples of two Integers, (optional, default : 
                ((0, 34), (160, 160)) parameters of the cropping process in
                shape ((xoffset, yoffset), (width, heigth))      
            compileKwargs : keyword arguments passed to the keras.Model.compile

        """

        # Create and compile model
        compileKwargs = kwargs.copy()
        if 'croppingParams' in compileKwargs:
            compileKwargs.pop('croppingParams')
        if 'frameSize' in compileKwargs:
            compileKwargs.pop('frameSize')
        self.model = keras.Model(inputs=inputs, outputs=layerStack, name=modelName)
        self.model.compile(loss=loss, optimizer=optimizer, **compileKwargs)

        # Define state's shape for image represented states ...
        if len(self.model.input_shape[2:]) in (2, 3):
            self.stateShape = kwargs.get('frameSize', np.array([84, 84]))
        # ... or fo other states types
        else:
            self.stateShape = self.model.input_shape[2:]
        # Define number of possible actions 
        self.actionsNum = self.model.output_shape[-1]
        # Save number of subsequent states that are combined to make a single agent's state
        self.stackedStateLength = stackedStateLength

        # Number of observations agent has made (observations are made only during training)
        self.observationsSeen = 0

        # Queue of the states received on actions (calling act(...) method)
        self.agentStateInitialized = False
        self.agentState = np.empty(np.concatenate((np.array([self.stackedStateLength]), self.stateShape), axis=0))

        # Internal replay memory used to store observation
        self.replayMemory = self.ReplayMemory(
            self.stateShape, stackedStateLength=stackedStateLength,
            size=memSize, batchSize=batchSize, stateDtype=stateDtype
        )

        # Action parameters
        self.__lastAction = 0
        self.__frameKeepCounter = 0

        # Discount rate parameters
        self.gammaPolicy = gammaPolicy
        # Epsilon policy parameters
        self.epsilonPolicy = epsilonPolicy
        # Size of the  single training mini-batch
        self.batchSize = batchSize
        # Pre-allocate memory for the training targets
        self.trainingTargets = np.empty((self.batchSize, self.actionsNum), dtype=stateDtype)


        # Optional image preprocessor
        if len(self.model.input_shape[2:]) in (2,3):
            self.__imagePreprocesor = self.ImagePreprocesor(
                imageShape=kwargs.get('imageShape', (84,84)),
                crop=kwargs.get('crop', ((0, 34), (160, 160)))
            )
        else:
            self.__imagePreprocesor = None




    def observe(self, action, reward, nextState, done):
        
        """
        Preprocess and saves single transition (composed of the state, action, reward
        and the next state) data to the internal log

        Args:
            action : Integer, action take in the iteration
            reward : float, reward obtained in the iteration
            next_state :  state observed after taking an action
            done : true if iteration was last in the episode

        Note:
            Actual state of the agent is hold internally

        """

        # If image-represented state, preprocess
        if self.__imagePreprocesor is not None:
            nextState = self.__imagePreprocesor(nextState)

        # Store observation
        self.replayMemory.store(action, reward, nextState, done )

        # Increment observation counter
        self.observationsSeen += 1




    def act(self, state, evaluation=False, frameKeep=1):
        
        """
        Returns action choosen for the given state

        Args:
            state : np.array, actual state
            evaluation : Boolean, true if agent is evaluated, False
                if trained
            frameKeep : Integer, number of iterations that a single
                action made by the agent should be preserved

        """    

        # Keep frame value has to be positive
        if frameKeep < 1:
            raise ValueError("'frameKeep' factor has to be positive!")

        # Preprocess state if it's an image-type
        if self.__imagePreprocesor is not None:
            state = self.__imagePreprocesor(state)

        # Initialize agent's state
        if not self.agentStateInitialized:
            self.agentState = np.repeat(
                state.reshape(np.concatenate((np.array([1]), state.shape), axis=0)),
                self.stackedStateLength, axis=0
            )
            self.agentStateInitialized = True
        # Update state
        else:
            self.agentState[:-1] = self.agentState[1:]
            self.agentState[-1] = state.reshape(np.concatenate((np.array([1]), state.shape), axis=0))

        # Make action
        if self.__frameKeepCounter == 0:

            # training actions
            if not evaluation:

                # Make random action with some probability
                if np.random.rand() <= self.epsilonPolicy(self.observationsSeen):
                    self.__lastAction = random.randrange(self.actionsNum)
                # Follow model's deems
                else:
                    self.__lastAction = np.argmax(self.model(
                        self.agentState.reshape(
                            np.concatenate((np.array([1]), self.agentState.shape), axis=0)
                        )
                    ).numpy()[0])

            # Evaluation actions
            else:
                self.__lastAction = np.argmax(self.model(
                    self.agentState.reshape(
                        np.concatenate((np.array([1]), self.agentState.shape), axis=0)
                    )
                ).numpy()[0])

        self.__frameKeepCounter += 1
        self.__frameKeepCounter %= frameKeep

        return self.__lastAction


    def stateReset(self):

        """
        Resets internal agent's state. This should be called between
        subsequent games.
        """

        self.agentStateInitialized = False




    def learn(self, **kwargs):


        """
        Perform a single mini-batch training basing on the random transitions
        sampled from the reply memory.

        Args : 
            batchSize : size of the single batch, if set of the saved transitions
                is not divisible by the batch size, the last batch is taken from
                remaining samples

        Kwargs : 
            arguments passed to the keras.Model.fit(...)

        Returns
            history dictionary returned by the keras.Model.fit

        """


        # Get transitions from the memory pool
        states, actions, rewards, nextStates, dones = self.replayMemory.getMiniBatch()

        # Prepare data for training
        for i in range(self.batchSize):

            # State target reward for the transition
            targetReward = rewards[i]
            if not dones[i]:
                targetReward = rewards[i] + self.gammaPolicy(self.observationsSeen) * np.amax(self.model(nextStates[i:i+1]).numpy()[0]) 

            # Compute network's target
            target = self.model(states[i:i+1]).numpy()
            target[0][actions[i]] = targetReward

            # Save training sample
            self.trainingTargets[i] = target

        # Perform batch
        return self.model.fit(states, self.trainingTargets, batch_size=self.batchSize, epochs=1, **kwargs)
    




    """
    Args:
        name : string, path to the file to save to
    """
    def load(self, name):
        self.model = keras.models.load_model(name)

    """
    Args:
        name : string, path to the file to load from    
    """
    def save(self, name):
        self.model.save(name)
