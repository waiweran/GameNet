# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import subprocess
import json
import os

class GainNet:

    def __init__(self, file=None):
        self.input_size = 87
        self.output_size = 18
        self.input_scale = 0.01
        self.test_fraction = 0.002
        self.epochs = 5
        self.raw_name = "checkpoints/adversarial_raw.h5"
        self.good_name = "checkpoints/adversarial_good.h5"

        if file:
            self.model = keras.models.load_model(file)
            self.model._make_predict_function()
        else:
            self._build_model()

    def _build_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(50, activation=tf.nn.relu, input_shape=(self.input_size,)),
            keras.layers.Dense(self.output_size, activation=tf.nn.softmax)
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def predict(self, state):
        scaled_state = state * self.input_scale
        output = self.model.predict(scaled_state)
        return output[0,:] # return weighted actions

    def train(self, input_data, target_data):
        train_data = input_data[0:int(len(input_data)*self.test_fraction)]
        test_data = input_data[int(len(input_data)*self.test_fraction):]
        train_targets = target_data[0:int(len(target_data)*self.test_fraction)]
        test_targets = target_data[int(len(target_data)*self.test_fraction):]
        train_data = train_data * self.input_scale
        test_data = test_data * self.input_scale
        self.model.fit(train_data, train_targets, epochs=self.epochs, validation_data=(test_data, test_targets))
        self.model.save(self.raw_name)
        if self.test():
            self.model.save(self.good_name)

    def test(self):
        basic_test = subprocess.run(['java', '-jar', 'Simulator.jar', '1000', 'WebRaw', 'BigMoney'])
        basic_comp = subprocess.run(['java', '-jar', 'Simulator.jar', '1000', 'WebGood', 'BigMoney'])
        net_test = subprocess.run(['java', '-jar', 'Simulator.jar', '1000', 'WebRaw', 'WebGood'])
        basic_result = basic_test.stdout.decode().splitlines()[-1]
        comp_result = basic_comp.stdout.decode().splitlines()[-1]
        net_result = net_test.stdout.decode().splitlines()[-1]
        print(basic_result)
        print(comp_result)
        print(net_result)

# Configuration Values
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
_,_,gain_data,gain_target = readFiles(datafiles, data_path, max_size=501000)

# Create Net
print("Creating Net")
agent = GainNet()

# Train and Test
print("Training")
agent.train(gain_data, gain_target)


