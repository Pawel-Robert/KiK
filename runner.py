from KiK.experience_buffer import ExperienceBuffer
from KiK.metrics import batch_metrics


class Runner:
    def __init__(self, board_size, agent_class, model, env, buffer_size, time_limit):
        """The main class to run the training loop"""
        self.board_size = board_size
        self.agent = agent_class(self.model, self.board_size)
        self.model = model
        self.env = env
        self.time_limit = time_limit

        self.buffer = ExperienceBuffer(buffer_size)

    def run_one_episode(self):
        """Plays one game and returns trajectory"""
        trajectory = None
        return trajectory

    def run_batch_of_episodes(self, n_episodes):
        """This could be parallelized to run on many cpus"""
        trajectory_batch = [self.run_one_episode() for _ in range(n_episodes)]
        return trajectory_batch

    def run(self, n_iterations, episodes_in_batch, data_size, epochs):
        """Full RL training loop"""
        for num in range(n_iterations):
            print(f'Processing step = {num}')
            trajectory_batch = self.run_batch_of_episodes(episodes_in_batch)
            print(batch_metrics(trajectory_batch, num))
            for trajectory in trajectory_batch:
                self.buffer.add_trajectory(trajectory)
            x, y = self.buffer.prepare_training_data(data_size)
            train_metrics = self.model.train(x, y, epochs)
            print(train_metrics)

        print(f'Training finished.')







        