""" Class representing memory gathered during plays, collected for training. """

import numpy as np

class ExperienceBuffer:
    """ Class representing memory gathered during plays, collected for training."""
    def __init__(self, buffer_size, alpha=0.5, gamma=0.95):
        """Here create a data structure to store trajectories, e.g. list, dictionary etc."""
        self.data = []
        """Parameters."""
        self.buffer_size = buffer_size
        self.lengths_of_trajectories = 0
        self.alpha = alpha
        self.gamma = gamma

    def add_trajectory(self, trajectory, algorithm, target_network):
        """Adds trajectory to buffer (trajectory = [state, action, q_value, reward, done]). """
        self.lengths_of_trajectories += len(trajectory)-1
        for i in range(len(trajectory)-1):
            current = trajectory[i]
            current_value = current[2]
            nxt = trajectory[i + 1]
            nxt_value = target_network.evaluate(nxt[0], nxt[1])
            q_value = np.array([algorithm.q_value(current_value, nxt_value, nxt[3], nxt[4],
                                                  self.alpha, self.gamma)])
            state = current[0]
            action = np.zeros(9)
            action[current[1]] = 1
            self.data.append([[state, action], q_value])

    def prepare_training_data(self):
        """Here we calculate targets for neural networks, i.e. pairs (x, y) to train on."""
        states_and_actions = [np.array([x[0][0].flatten() for x in self.data]),
                              np.array([x[0][1] for x in self.data])]
        values = np.array([x[1] for x in self.data])
        return states_and_actions, values

    def clear_buffer(self):
        """ Clears first data_to_clear entries in self.data. """
        self.data = []
