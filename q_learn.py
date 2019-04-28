from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
from collections import deque
from game_env import Dominion
import numpy as np
import random
import os

EPISODES = 20000

class GainDQN:

    def __init__(self, file=None, epsilon=1.0):
        self.input_size = 87
        self.output_size = 18
        self.input_scale = 0.01
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        if file:
            self.model = self._load_model(file)
        else:
            self.model = self._build_model()

    def _load_model(self, file):
        model = keras.models.load_model(file)
        model._make_predict_function()
        return model

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(50, input_dim=self.input_size, activation=tf.nn.relu))
        # model.add(keras.layers.Dense(20, activation=tf.nn.relu))
        # model.add(keras.layers.Dense(20, activation=tf.nn.relu))
        model.add(keras.layers.Dense(self.output_size, activation=tf.nn.softmax))
        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        if np.random.rand() <= self.epsilon:
            return ["random"]
        scaled_state = state * self.input_scale
        output = self.model.predict(scaled_state)
        return output[0,:] # return weighted actions

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                scaled_next_state = next_state * self.input_scale
                target = (reward + self.gamma * np.amax(self.model.predict(scaled_next_state)[0]))
            scaled_state = state * self.input_scale
            target_f = self.model.predict(scaled_state)
            target_f[0][action] = target
            self.model.fit(scaled_state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_weights(self, name):
        self.model.load_weights(name)

    def save_weights(self, name):
        self.model.save_weights(name)

    def save(self, name):
        self.model.save(name)



if __name__ == "__main__":
    agent = GainDQN(file='checkpoints/gain-pretrain.h5', epsilon=0.01)
    env = Dominion()
    
    done = False
    batch_size = 200

    for e in range(EPISODES):
        state = env.reset()
        state = np.expand_dims(state, 0)
        while True:
            prediction = agent.predict(state)
            next_state, reward, done, action, score = env.step(prediction)
            next_state = np.expand_dims(next_state, 0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, score, agent.epsilon))
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                break
        if e > 0 and e % 100 == 0:
            agent.save("checkpoints/gain-dqn-{}.h5".format(e))

    agent.save("checkpoints/gain-dqn.h5")
    env.close()
