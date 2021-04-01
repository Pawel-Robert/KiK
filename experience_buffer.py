# from numpy import random
import numpy as np

class ExperienceBuffer:
    def __init__(self, buffer_size, alpha=0.8, gamma=0.95, sort='best'):
        self.buffer_size = buffer_size
        self.sort = sort
        # Here create a data structure to store trajectories, e.g. list, dictionary etc.
        # self.trajectories = []
        self.data = []
        self.alpha = alpha
        self.gamma = gamma

    def add_trajectory(self, trajectory):
        """Adds trajectory to buffer"""
        # lista pomocnicza
        current = trajectory[0]
        # pojedyncza trajektoria na postać [state, action, q_value, reward, done]
        # state = trajectory[0]
        # action = trajectory[1]
        # q_value = trajectory[2]
        # reward = trajectory[3]
        # done = trajectory[4]
        for next in trajectory[1:]:

            # obrabiamy trochę dane
            st = current[0]
            st = st.flatten()
            st_input = np.array([st])
            ac = np.zeros(9)
            action = current[1]
            ac[action] = 1
            ac_input = np.array([ac])

            # aby pozbyć się problemu z kopiowaniem
            target_Q_value = 0
            Q_value = 0

            # obliczamy wartość docelową Q dla treningu
            # musimy propagować zwycięstwo wstecz, uwzględniając potęgi gamma
            target_Q_value = self.alpha*current[2] + (1-self.alpha)*next[2] + current[3]*self.gamma
            Q_value = np.copy(target_Q_value[0][0])

            # aktualizujemy buffer
            self.data.append([[st_input, ac_input], Q_value])

            # zachowujemy do kolejnego kroku jako wartość poprzednią
            current = next
        pass

    def prepare_training_data(self, data_size):
        """Here we calculate targets for naural networks, i.e. pairs (x, y) to train on."""
        # bierzemy określoną ilosć losowych próbek z trajektorii (aby uniknąć przeładowania)
        # x to aktualny stan gry, a y to wartość Q lepsza niż model(x), uzyskana z równania Bellmana
        # trajectory to lista czwórek postaci: (observation, action, reward, done)
        st = [[] for i in range(len(self.data))]
        ac = [[] for i in range(len(self.data))]
        y = np.zeros(len(self.data))

        # losujemy n=data_size próbek par treningowych i w odpowiednim formacie przesyłamy je dalej
        for i in range(len(self.data)):
            #sample = np.random.randint(data_size)
            #sample = i
            st[i] = self.data[i][0][0][0]
            ac[i] = self.data[i][0][1][0]
            target_Q_value = self.data[i][1]
            y[i] = target_Q_value
        x_1 = np.stack(st)
        x_2 = np.stack(ac)
        x = [x_1, x_2]
        return x, y


    def clear_buffer(self):
        self.trajectories = []
        pass






