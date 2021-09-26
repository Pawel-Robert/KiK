""" Network classes. """

from keras.layers import Input, Dense, Concatenate, Conv2D, MaxPool2D, Flatten
from keras.models import Model
import numpy as np


class ValueNetwork:
    """ Q value network class. """
    def __init__(self, height, width):
        self.height = height
        self.width = width

        input_state = Input(shape=(self.height, self.width,1,))
        state = Conv2D(filters=50, kernel_size=(5, 5), activation='relu')(input_state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Conv2D(filters=50, kernel_size=(3, 3), activation='relu')(state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Flatten()(state)
        layer = Dense(800, activation='relu')(state)
        layer = Dense(100, activation='relu')(layer)
        value = Dense(1)(layer)

        self.model = Model(inputs=[input_state], outputs=[value])
        self.model.compile(loss='mse', metrics=['mse'])

    def evaluate(self, state, action):
        """ Returns the output of the network for specific state and action. """
        state[action // self.width, action % self.width] = 1
        return self.model([np.array([state])])[0][0]

    def evaluate_on_batch(self, state, list_of_actions):
        """ Returns the output of the network for specific state and a list of actions. """
        list_of_states = []
        for action in list_of_actions:
            temp_state = np.copy(state)
            temp_state[action // self.width, action % self.width] = 1
            list_of_states.append(temp_state)
        return self.model.predict_on_batch([np.array(list_of_states)])


class PolicyNetwork:
    """ Policy network used to sample actions for QNetwork. """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        input_state = Input(shape=(self.height, self.width,1,))
        state = Conv2D(filters=26, kernel_size=(3, 3), activation='relu')(input_state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Conv2D(filters=36, kernel_size=(3, 3), activation='relu')(state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Flatten()(state)
        layer = Dense(800, activation='relu')(state)
        layer = Dense(100, activation='relu')(layer)
        logits = Dense(self.width * self.height, activation='softmax')(layer)

        self.model = Model(inputs=[input_state], outputs=[logits])
        self.model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy'])

    def evaluate(self, state):
        """ Returns the output of the network for specific state and action. """
        return self.model([np.array([state])])
