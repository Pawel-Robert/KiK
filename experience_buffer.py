
import numpy as np

""" Class representing memory gathered during plays, collected for training."""
class ExperienceBuffer:
    def __init__(self, buffer_size, height, width, target_network_model, alpha=0.8, gamma=0.95, sort='best'):
        self.buffer_size = buffer_size
        self.sort = sort
        """ Used to improve DQN algorithm. """
        self.target_network_model = target_network_model
        """Parameters."""
        self.alpha = alpha
        self.gamma = gamma
        self.height = height
        self.width = width
        """Here create a data structure to store trajectories, e.g. list, dictionary etc."""
        self.data = []
        """ Number of inputs in self.data, which measures training samples in the last trajectory batch.
            It is used in function clear buffer to forget about old enought samples. """
        self.lengths_of_trajectories = 0

    # def add_MCTS_trajectory(self, trajectory):
    #     """ Use MCTS algorithm to compute target Q value for the learining process. """
    #     """ Rise the length of trajectories. """
    #     self.lengths_of_trajectories += len(trajectory) - 1
    #
    #     """trajectory = [state_1, state_2, action, q_value, reward, done]"""
    #
    #     current = trajectory[0]
    #
    #     for next in trajectory[1:]:
    #         """Prepare data for training."""
    #         st_1 = current[0]
    #         st_1 = st_1.flatten()
    #         st_input_1 = np.array([st_1])
    #         st_2 = current[1]
    #         st_2 = st_2.flatten()
    #         st_input_2 = np.array([st_2])
    #         ac = np.zeros(9)
    #         action = current[2]
    #         ac[action] = 1
    #         ac_input = np.array([ac])
    #         Q_value = 0
    #         """ In the case of winning we can set a specific value of the target Q value. """
    #         if next[5]:
    #             Q_value = self.gamma * next[4]
    #             """ If it is not the final step we use MCTS algorithm to upgread the Q value. """
    #         else:
    #             state = st_1 + st_2
    #             mcst.predict(state)
    #             Q_value = self.alpha * current[3] + (1 - self.alpha) * next[3]
    #         Q_value = np.array([Q_value])
    #         # if current[4] == True:
    #         #     print(next[3])
    #         #     print(f' current {current[4]}')
    #         #     print(Q_value)
    #         self.data.append([[st_input_1, st_input_2, ac_input], Q_value])
    #
    #         """Save current value for the next step."""
    #         current = next
    #     pass




    def add_trajectory(self, trajectory):
        """Adds trajectory to buffer"""


        """ Rise the length of trajectories. """
        self.lengths_of_trajectories += len(trajectory)-1

        """trajectory = [state_1, state_2, action, q_value, reward, done]"""

        "Initial values of data from the trajectory to be consumed by the network."
        current = trajectory[0]
        st_1 = current[0]
        st_1 = st_1.flatten()
        st_input_1 = np.array([st_1])
        st_2 = current[1]
        st_2 = st_2.flatten()
        st_input_2 = np.array([st_2])
        ac = np.zeros(9)
        action = current[2]
        ac[action] = 1
        ac_input = np.array([ac])

        for next in trajectory[1:]:

            """Prepare data from the trajectory to be consumed by the target network."""
            n_st_1 = next[0]
            n_st_1 = n_st_1.flatten()
            n_st_input_1 = np.array([n_st_1])
            n_st_2 = next[1]
            n_st_2 = n_st_2.flatten()
            n_st_input_2 = np.array([n_st_2])
            n_ac = np.zeros(9)
            n_action = next[2]
            n_ac[n_action] = 1
            n_ac_input = np.array([n_ac])

            Q_value = 0
            """ In the case of the last move we know the exact value of the Q-function. """
            if next[5]:
                Q_value = self.gamma * next[4]
                # print(f'final {Q_value}')
                """ Otherwise we use Bellman equation to compute the target value of the Q-function. """
                """ Target value of the Q-function is lagged, when compared with the learned Q-function. """
                """ We need to change the value of the next[3], using target network. """
            else:
                target_value = float(self.target_network_model([n_st_input_1, n_st_input_2, n_ac_input])[0][0])
                Q_value = self.alpha * current[3] + (1-self.alpha) * target_value
            Q_value = np.array([Q_value])
            # if current[4] == True:
            #     print(next[3])
            #     print(f' current {current[4]}')
            #     print(Q_value)
            self.data.append([[st_input_1, st_input_2, ac_input], Q_value])

            """Actualise various parameters."""
            current = next
            st_input_1 = n_st_input_1
            st_input_2 = n_st_input_2
            ac_input = n_ac_input
        pass

    def prepare_training_data(self, data_size):
        """Here we calculate targets for neural networks, i.e. pairs (x, y) to train on."""


        st_1 = [[] for i in range(len(self.data))]
        st_2 = [[] for i in range(len(self.data))]
        ac = [[] for i in range(len(self.data))]
        y = np.zeros(len(self.data))

        for i in range(len(self.data)):
            st_1[i] = np.copy(self.data[i][0][0][0])
            st_2[i] = np.copy(self.data[i][0][1][0])
            ac[i] = np.copy(self.data[i][0][2][0])
            target_Q_value = self.data[i][1]
            y[i] = target_Q_value

        x_1 = np.stack(st_1)
        x_2 = np.stack(st_2)
        x_3 = np.stack(ac)
        x = [x_1, x_2, x_3]
        # print(self.data)
        return x, y

    def clear_buffer(self):
        # print(f'Długość trajektorii = {self.lengths_of_trajectories}')
        # print(f'len self data = {len(self.data)}')
        self.data = self.data[self.lengths_of_trajectories:]
        # print(f'len self data = {len(self.data)}')
        self.lengths_of_trajectories = 0
        pass








