class ExperienceBuffer:
    def __init__(self, buffer_size, sort='best'):
        self.buffer_size = buffer_size
        self.sort = sort
        # Here create a data structure to store trajectories, e.g. list, dictionary etc.

    def add_trajectory(self, trajectory):
        """Adds trajectory to buffer"""
        raise NotImplementedError

    def prepare_training_data(self, data_size):
        """Here we calculate targets for naural networks, i.e. pairs (x, y) to train on."""
        raise NotImplementedError

    def clear_buffer(self):
        raise NotImplementedError
