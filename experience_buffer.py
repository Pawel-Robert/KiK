from numpy import random

class ExperienceBuffer:
    def __init__(self, buffer_size, sort='best'):
        self.buffer_size = buffer_size
        self.sort = sort
        # Here create a data structure to store trajectories, e.g. list, dictionary etc.
        # self.trajectories = []
        self.data = [[],[]]

    def add_trajectory(self, trajectory):
        """Adds trajectory to buffer"""
        # self.trajectories.append(trajectory)
        # lista pomocnicza
        b = []
        gamma = 0.95
        # a = (observation, action, reward, done)
        for a in trajectory:
            if b != []:
                # stan poprzedni
                self.data[0].append(b[0])
                # TODO: tu użyć równania Bellmana
                target_Q_value = a[3] + a[1]*gamma
                self.data[1].append(target_Q_value)
            # zachowujemy do kolejnego kroku jako wartość poprzednią
            b = a
        pass

    def prepare_training_data(self, data_size):
        """Here we calculate targets for naural networks, i.e. pairs (x, y) to train on."""
        # bierzemy określoną ilosć losowych próbek z trajektorii (aby uniknąć przeładowania)
        # x to aktualny stan gry, a y to wartość Q lepsza niż model(x), uzyskana z równania Bellmana
        # trajectory to lista czwórek postaci: (observation, action, reward, done)
        samples = random.sample(self.data, data_size)
        x = []
        y = []
        for sample in samples:
            state, target_Q_value = sample
            x.append(state)
            y.append(target_Q_value)
        return x, y


    def clear_buffer(self):
        self.trajectories = []
        pass






