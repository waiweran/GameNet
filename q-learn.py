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

EPISODES = 1000

class GainDQN:

    def __init__(self):
        self.input_size = 87
        self.output_size = 18
        self.input_scale = 0.01
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(50, input_dim=self.input_size, activation=tf.nn.relu))
        model.add(keras.layers.Dense(20, activation=tf.nn.relu))
        model.add(keras.layers.Dense(20, activation=tf.nn.relu))
        model.add(keras.layers.Dense(self.output_size, activation=tf.nn.softmax))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.dirichlet(np.ones(self.output_size)/10., size=1)
        scaled_state = state * self.input_scale
        output = self.model.predict(scaled_state)
        return output # return weighted actions

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    agent = GainDQN()
    env = Dominion()
    
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.expand_dims(state, 0)
        while True:
            prediction = agent.predict(state)
            next_state, reward, done, action, score = env.step(prediction[0,:])
            next_state = np.expand_dims(next_state, 0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, score, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    agent.save("checkpoints/gain-dqn.h5")
    env.close()