""" Network class. """

from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import numpy as np

class QValue:
    """ Network class. """
    def __init__(self):
        input_state = Input(shape=(9,))
        input_action = Input(shape=(9,))

        state_and_action = Concatenate()([input_state, input_action])
        layer = Dense(36, activation='relu')(state_and_action)
        layer = Dense(36, activation='relu')(layer)
        q_value = Dense(1)(layer)

        self.model = Model(inputs=[input_state, input_action], outputs=[q_value])
        self.model.compile(loss='mse', metrics=['mse'])

    def evaluate(self, state, action):
        """ Returns the output of the network for specific state and action. """
        action_array = np.zeros(9)
        action_array[action] = 1
        return self.model([np.array([state.flatten()]), np.array([action_array])])[0][0]
