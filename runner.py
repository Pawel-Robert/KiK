from experience_buffer import ExperienceBuffer
from metrics import batch_metrics


class Runner:
    def __init__(self, agent_class, model, env, buffer_size, time_limit):
        """The main class to run the training loop"""
        # tworzymy agenta o typie podanym w parametrze agent_class
        self.model = model
        self.agent = agent_class(self.model)
        self.env = env
        self.time_limit = time_limit

        self.buffer = ExperienceBuffer(buffer_size)

    def run_one_episode(self):
        """Plays one game and returns trajectory"""
        trajectory = None
        time = 0
        observation = self.env.reset()
        while True:
            # agent wybiera akcję na polu jeszcze nie zajętym
            while True:
                action = self.agent.act(observation)
                if self.env.is_allowed_move(action):
                    break
            # wybraną akcją wpływamy na środowiski i zbieramy obserwacje
            next_observation, reward, done, info = self.env.step(action)
            # obserwacje dodajemy do trajektorii
            trajectory.append(observation, action, reward, done)
            if done:
                break
            # update our observatons
            observation = next_observation
            # sprawdzmy, czy nie przekroczyliśmy limitu czasu
            time += 1
            if time == self.time_limit:
                break
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
            train_metrics = self.model.fit(x, y, epochs)
            print(train_metrics)

        print(f'Training finished.')







        