from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import json

def load_model()
	model = keras.models.load_model('checkpoints/gain.h5')
	return model


def run_game(model, input):
	indict = json.loads(instring)
	netin = (np.expand_dims(indict['GainChoice'], 0))
	output = model.predict(netin)
	output = output.tolist()[0]
	return json.dumps(output)

if __name__ == '__main__':
	model = load_model()
	print(model.input_shape)
	model.summary()
	while True:
		instring = input()
		print(run_game(model, instring))




