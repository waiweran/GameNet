from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import json
import os

# Configuration Values
test_fraction = 9/10
scale_factor = 100
epochs = 10
data_path = "Training"

# Reads and combines a set of data files
def readFiles(files, directory):
	playData = list()
	playTarget = list()
	gainData = list()
	gainTarget = list()

	for file in files:
		with open(os.path.join(directory, file)) as datafile:
			data = json.loads(datafile.read())
			playData.extend(data['playData'])
			playTarget.extend(data['playTarget'])
			gainData.extend(data['gainData'])
			gainTarget.extend(data['gainTarget'])
		if len(gainTarget) > 10000:
			break

	return np.array(playData), np.array(playTarget), np.array(gainData), np.array(gainTarget)

# Read Data Files
print("Listing Files")
files = os.listdir(data_path)
datafiles = list()
for file in files:
	if file.endswith(".json"):
		datafiles.append(file)
print("Loading Files")
train_files = datafiles[0:int(len(datafiles)*test_fraction)]
test_files = datafiles[int(len(datafiles)*test_fraction):]
play_train_data,play_train_target,gain_train_data,gain_train_target = readFiles(train_files, data_path)
play_test_data,play_test_target,gain_test_data,gain_test_target = readFiles(test_files, data_path)

# Scale Inputs
print("Scaling Inputs")
play_train_data = play_train_data / scale_factor
play_test_data = play_test_data / scale_factor
gain_train_data = gain_train_data / scale_factor
gain_test_data = gain_test_data / scale_factor

# Create Net
print("Creating Net")
model = keras.Sequential([
    keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(87,)),
    keras.layers.Dense(18, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and Test
print("Training")
checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoints/gain.h5", save_weights_only=False, period=5)
model.fit(gain_train_data, gain_train_target, epochs=epochs,
	      validation_data=(gain_test_data, gain_test_target), callbacks=[checkpoint])

model.summary()
