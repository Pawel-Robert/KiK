class ExperienceBuffer:
    def __init__(self, buffer_size, sort='best'):
        self.buffer_size = buffer_size
        self.sort = sort
        # Here create a data structure to store trajectories, e.g. list, dictionary etc.
        self.observations = []
        self.actions = []
        self.rewards = []

    def add_trajectory(self, trajectory):
        """Adds trajectory to buffer"""
        self.trajectories.append(trajectory)

    def prepare_training_data(self, data_size):
        """Here we calculate targets for naural networks, i.e. pairs (x, y) to train on."""
        raise NotImplementedError

    def clear_buffer(self):
        self.trajectories = []

