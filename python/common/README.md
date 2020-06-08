# Description of the configuration parameters required by the session()
{
    "env" : "Gym ID of the used environment",
    "time" : {
        "maxFramesNum" : "maximum number of games' iterations (i.e. env.step() calls)",
        "maxFramesPerGame" : "maximum number of the single game's iterations",
        "learningFrequency" : "number of training frames between agent.learn() calls",
        "evaluationFreq" : "number of training frames between evaluation",
        "evaluationGames" : "number of games played in the evaluation step"
    },
    "agent" : {
        "initialRandomFrames" : "number of initial training frames when agent doesn't learn ",
        "clipReward" : "if true, positive rewards are normalized to '1', and negative to '-1'",
        "trainingFrameKeep" : "number of frames that agent holds action unchanged (at training)",
        "evaluationFrameKeep" : "number of frames that agent holds action unchanged (at evaluation)"
    },
    "log: : {
        "directory" : "path to the folder with Tensorboard logs",
        "logID" : "ID of the log folder",
        "display" : "if true, training frames are rendered",
        "displayEval" : "if true, evaluation frames are rendered",
        "meanTrainingScoreLength" : "number of recent games that mean training score is calculated on",
        "verboseLearning" : "if true, keras.Model.fit() is called with the verbose=1",
        "trainingLogUpdateFreq" : "number of training games between updating Tensorboard"
    },
    "environment" : {
        "evaluationRandomStart" : "maximum number of random actions taken at the each evaluation game",
        "evaluationRandomStartAction" : "identifier of the action to be performed at the begining of the evaluation game",
        "infoAsDone" : "In some games (e.g. Breakout) single game consists of single episodes. If this parameters is set
                        to false, game is considered 'done' when 'done' returned from env.step() is True. This field can
                        be set to the key-string that determines field of the 'info' dictionary that servers this way
                        instead (e.g. you can consider lost of the each life as a terminal state)"
    }
}

# Tensorboard
When session() method is called, the tensorboard host can be activated with tensorboard --logdir="directory/logID"