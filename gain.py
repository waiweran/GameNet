from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import json

class GainModel:

	def __init__(self):
		self.model = keras.models.load_model('checkpoints/gain.h5')

	def run_game(self, input):
		indict = json.loads(input)
		netin = (np.expand_dims(indict['GainChoice'], 0))
		output = self.model.predict(netin)
		output = output.tolist()[0]
		return json.dumps(output)


if __name__ == '__main__':
	gain = GainModel()
	print(gain.model.input_shape)
	gain.model.summary()
	while True:
		instring = input()
		print(run_game(instring))




