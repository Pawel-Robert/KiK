
import numpy as np

""" Class representing memory gathered during plays, collected for training."""
class ExperienceBuffer:
    def __init__(self, buffer_size, height, width, alpha=0.8, gamma=0.95, sort='best'):
        self.buffer_size = buffer_size
        self.sort = sort
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



    def add_trajectory(self, trajectory):
        """Adds trajectory to buffer"""


        """ Rise the length of trajectories. """
        self.lengths_of_trajectories += len(trajectory)-1

        """trajectory = [state, action, q_value, reward, done]"""

        current = trajectory[0]

        for next in trajectory[1:]:
            """Prepare data for training."""
            st = current[0]
            st = st.flatten()
            st_input = np.array([st])
            ac = np.zeros(9)
            action = current[1]
            ac[action] = 1
            ac_input = np.array([ac])
            Q_value = 0
            if next[4]:
                Q_value = self.gamma * next[3]
                # print(f'final {Q_value}')
            else:
                Q_value = self.alpha * current[2] + (1-self.alpha) * next[2]
            Q_value = np.array([Q_value])
            # if current[4] == True:
            #     print(next[3])
            #     print(f' current {current[4]}')
            #     print(Q_value)
            self.data.append([[st_input, ac_input], Q_value])

            """Save current value for the next step."""
            current = next
        pass

    def prepare_training_data(self, data_size):
        """Here we calculate targets for neural networks, i.e. pairs (x, y) to train on."""


        st = [[] for i in range(len(self.data))]
        ac = [[] for i in range(len(self.data))]
        y = np.zeros(len(self.data))

        for i in range(len(self.data)):
            st[i] = np.copy(self.data[i][0][0][0])
            ac[i] = np.copy(self.data[i][0][1][0])
            target_Q_value = self.data[i][1]
            y[i] = target_Q_value

        x_1 = np.stack(st)
        x_2 = np.stack(ac)
        x = [x_1, x_2]
        # print(self.data)
        return x, y

    def clear_buffer(self):
        # print(f'Długość trajektorii = {self.lengths_of_trajectories}')
        # print(f'len self data = {len(self.data)}')
        self.data = self.data[self.lengths_of_trajectories:]
        # print(f'len self data = {len(self.data)}')
        self.lengths_of_trajectories = 0
        pass

    def add_trajectory_general(self, trajectory):
        """Adds trajectory to buffer"""

        """ Rise the length of trajectories. """
        self.lengths_of_trajectories += len(trajectory) - 1

        """trajectory = [state, action, q_value, reward, done]"""

        current = trajectory[0]

        for next in trajectory[1:]:
            """Prepare data for training."""
            st = current[0]
            st_input = np.array([st])
            action = current[1]
            ac = np.zeros((self.height, self.width))
            a = action % self.width
            b = action // self.width
            ac[a, b] = 1
            ac_input = np.array([ac])
            Q_value = 0
            if next[4]:
                Q_value = self.gamma * next[3]
                # print(f'final {Q_value}')
            else:
                Q_value = self.alpha * current[2] + (1 - self.alpha) * next[2]
            Q_value = np.array([Q_value])
            # if current[4] == True:
            #     print(next[3])
            #     print(f' current {current[4]}')
            #     print(Q_value)
            self.data.append([[st_input, ac_input], Q_value])

            """Save current value for the next step."""
            current = next
        pass

    def prepare_training_data_general(self, data_size):
        """Here we calculate targets for naural networks, i.e. pairs (x, y) to train on."""
        # bierzemy określoną ilosć losowych próbek z trajektorii (aby uniknąć przeładowania)
        # x to aktualny stan gry, a y to wartość Q lepsza niż model(x), uzyskana z równania Bellmana
        # trajectory to lista czwórek postaci: (observation, action, reward, done)

        st = [[] for i in range(len(self.data))]
        ac = [[] for i in range(len(self.data))]
        y = np.zeros(len(self.data))
        # print(self.data[0])
        # losujemy n=data_size próbek par treningowych i w odpowiednim formacie przesyłamy je dalej
        for i in range(len(self.data)):
            #sample = np.random.randint(data_size)
            #sample = i
            st[i] = np.copy(self.data[i][0][0][0])
            ac[i] = np.copy(self.data[i][0][1][0])
            target_Q_value = self.data[i][1]
            y[i] = target_Q_value
        x_1 = np.stack(st)
        x_2 = np.stack(ac)
        # print(x_1, x_2, y)
        x = [x_1, x_2]
        # print(x,y)
        # print(x[0],y)
        return x, y








