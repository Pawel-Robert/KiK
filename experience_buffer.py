""" Class representing memory gathered during plays, collected for training. """

import numpy as np


class ExperienceBuffer:
    """ Class representing memory gathered during plays, collected for training."""
    def __init__(self, buffer_size):
        """Here create a data structure to store trajectories, e.g. list, dictionary etc."""
        self.data = []
        """Parameters."""
        self.buffer_size = buffer_size

    def add_trajectory(self, trajectory, algorithm, network):
        """Adds trajectory to buffer (trajectory[i] = [state, action, q_value, reward, done]). """
        current = trajectory[0]
        for nxt in trajectory[1:]:
            current_value = current[2]
            state = current[0]
            state[current[1] // state.shape[0], current[1] % state.shape[1]] = 1
            nxt_value = network.evaluate(nxt[0], nxt[1])
            q_value = np.array([algorithm.q_value(current_value, nxt_value, nxt[3], nxt[4])])
            current = nxt
            self.data.append([state, q_value])

    def prepare_training_data(self):
        """Here we calculate targets for neural networks, i.e. pairs (x, y) to train on."""
        states = [np.array([x[0] for x in self.data])]
        values = np.array([x[1] for x in self.data])
        return states, values

    def clear_buffer(self):
        """ Clears first data_to_clear entries in self.data. """
        self.data = []

