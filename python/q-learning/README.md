# Descrption of the parameters specific to DQN (for base parameters, @see python/common/README.md)
{
    "computeDevices": "ID of the devices used for computation. "-1" for CPU, "0, 1, ..." for GPU ",
    "mode": "Mode of the next run ('avaluation' or 'training)"
    "paths": {
        "savesDir" : "path to the directory storing models and memory stamps",
        "initialModelName" : "name of the model save to load before training (it is supposed to be
                              on the 'savesDir/models/' directory). If false, new model is created",
        "initialReplayMemoryName" : "name of the memory stamp to load before training (it is supposed to be
                              on the 'savesDir/replays/' directory). If false, no memory is loaded",
    },
    "agent": {
        "replayMemorySize" : "maximum size of the replay memory"
        "initialRandomFrames*" : "In this context it is also size of the initial replay memory (i.e. number
                                  game transitions that are gathered in the replay memory before agent starts
                                  learning)",
        "stackedStateLength" : "number of subsequent frames that are stacked to create an agent's state",
        "epsilonPolicy" : "name of the policy for epsilon management ('linear' or 'exponential')",
        "epsilonPolicyConfig": {
            "linear": {
                "initial" : "initial epsilon (0 <= e <= 1)",
                "initialPeriod" : "number of frames that initial value is hold",
                "firstTarget" : "first target epsilon (0 <= e <= 1)",
                "firstTargetPeriod" : "number of frames within the first target is reached (from the 'initialPeriod' moment)",
                "finalTarget" : "final epsilon value (0 <= e <= 1)",
                "finalTargetPeriod" : "number of frames within the final target is reached (from reaching the first target)"
            },
            "exponential": {
                "initial" : "initial value of the epsilon",
                "decay" : "decay coefficient (e = initial^(decay)"
            }
        }
    },
    "model": {
        "optimizer" : "name of the keras optimizer"
        "lossFunction" : "name of the keras loss function",
        "learningRate" : "learning rate,
        "batchSize" : "size of the mini-batch",
        "layers" : [
            {
                "type" : "name of the keras layer (lowercase camelCase)",
                "\units\ : "number of units ( \dense\ )",
                "\activation\" : "name of the keras activation function",
                "\initializer\" : "name of the keras kernel initializer (lowercase camelCase)"
                "\scale\" : "scaling factor ( \variance-scaling initializer\ )"
            }
        ],
        "name" : "user's name of the keras model"
    }
}

NOTE: These \parameters\ are case-specific