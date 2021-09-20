""" Network class. """

from keras.layers import Input, Dense, Concatenate, Conv2D, MaxPool2D, Flatten
from keras.models import Model
import numpy as np

class QValue:
    """ Q value network class. """
    def __init__(self, height, width):
        self.height = height
        self.width = width

        input_state = Input(shape=(self.height, self.width,1,))
        state = Conv2D(filters=26, kernel_size=(3, 3), activation='relu')(input_state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Conv2D(filters=36, kernel_size=(3, 3), activation='relu')(state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Flatten()(state)

        input_action = Input(shape=(self.height, self.width,1,))
        action = Conv2D(filters=26, kernel_size=(3, 3), activation='relu')(input_action)
        action = MaxPool2D(pool_size=(2, 2))(action)
        action = Conv2D(filters=36, kernel_size=(3, 3), activation='relu')(action)
        action = MaxPool2D(pool_size=(2, 2))(action)
        action = Flatten()(action)

        state_and_action = Concatenate()([state, action])
        layer = Dense(100, activation='relu')(state_and_action)
        layer = Dense(100, activation='relu')(layer)
        q_value = Dense(1)(layer)

        self.model = Model(inputs=[input_state, input_action], outputs=[q_value])
        self.model.compile(loss='mse', metrics=['mse'])

    def evaluate(self, state, action):
        """ Returns the output of the network for specific state and action. """
        action_array = np.zeros((self.height, self.width))
        action_array[action // self.width, action % self.height] = 1
        return self.model([np.array([state]), np.array([action_array])])[0][0]

class ValueNetwork:
    """ Value network class. """

    def __init__(self, height, width):
        self.height = height
        self.width = width
        input_state = Input(shape=(height, width))
        state = Conv2D(20, 3, 1)(input_state)
        state = Conv2D(20, 3, 1)(state)
        state = Conv2D(20, 3, 1)(state)
        layer = Dense(100, activation='relu')(state)
        layer = Dense(100, activation='relu')(layer)
        logits = Dense(self.width * self.height, activation='softmax')(layer)

        self.model = Model(inputs=[input_state], outputs=[logits])
        self.model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_crossentropy'])

    def evaluate(self, state):
        """ Returns the output of the network for specific state and action. """
        return self.model([np.array([state])])
