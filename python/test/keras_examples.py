# Disable tf logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Basic imprts
import numpy as np
import tensorflow as tf
from tensorflow import keras

#-------------------------------------------------------------------------------------

# Keras imports
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import layers

# Image displaying
from PIL import Image

#-------------------------------------------------------------------------------------

# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(
  optimizer="adam",
  loss="sparse_categorical_crossentropy",
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

# Train the model for 1 epoch from Numpy data
batch_size = 64
callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs')
]
history = model.fit(
  x_train, y_train,
  batch_size=batch_size,
  epochs=1,
  callbacks=callbacks  
)

print(model.predict(x_test[0].reshape((1, 28, 28))))