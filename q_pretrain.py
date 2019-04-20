from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Neural net
from q_learn import GainDQN
from game_env import Dominion

# Helper libraries
import numpy as np
import json
import os

# Configuration Values
test_fraction = 9/10
epochs = 10
data_path = "Training"

# Reads and combines a set of data files
def readFiles(files, directory, max_size=0):
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
		if max_size > 0 and len(gainTarget) > max_size:
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
play_train_data,play_train_target,gain_train_data,gain_train_target = readFiles(train_files, data_path, max_size=5000)
play_test_data,play_test_target,gain_test_data,gain_test_target = readFiles(test_files, data_path, max_size=100)

# Create Net
print("Creating Net")
agent = GainDQN()
env = Dominion()

# Scale Inputs
print("Scaling Inputs")
play_train_data = play_train_data * agent.input_scale
play_test_data = play_test_data * agent.input_scale
gain_train_data = gain_train_data * agent.input_scale
gain_test_data = gain_test_data * agent.input_scale
gain_train_target_arr = np.zeros(shape=(len(gain_train_target), 18))
for i in range(len(gain_train_target)):
	gain_train_target_arr[i][gain_train_target[i]] = 1
gain_test_target_arr = np.zeros(shape=(len(gain_test_target), 18))
for i in range(len(gain_test_target)):
	gain_test_target_arr[i][gain_test_target[i]] = 1

print(np.shape(gain_train_data))
print(np.shape(gain_train_target_arr))

# Train
print("Training")
checkpoint = tf.keras.callbacks.ModelCheckpoint("checkpoints/gain_weights.h5", save_weights_only=True, period=5)
agent.model.fit(gain_train_data, gain_train_target_arr, epochs=epochs,
	      validation_data=(gain_test_data, gain_test_target_arr), callbacks=[checkpoint])

agent.save("checkpoints/gain_dqn_pretrain.h5")
agent.model.summary()

done = False
state = env.reset(100)
while not done:
    prediction = agent.predict(state)
    next_state, reward, done, action, score = env.step(prediction[0,:])
    state = np.expand_dims(next_state, 0)
env.close()
