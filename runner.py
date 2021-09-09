
from tqdm import tqdm
from experience_buffer import ExperienceBuffer
from numpy import copy, random
from datetime import datetime
from tester import Tester


class Runner:
    def __init__(self, agent_class, algorithm_class, network, env, epsilon=0.1, buffer_size=10000):
        """The main class to run the training loop"""
        self.network = network
        self.agent = agent_class(network, epsilon)
        self.env = env
        """ Initialise experience buffer with a target network. """
        self.buffer = ExperienceBuffer(buffer_size)
        self.tester = Tester(env)
        self.algorithm = algorithm_class()

    def run_one_episode(self, iteration):
        """Plays one game AGAINST ITSELF and returns a trajectory of one player."""
        trajectory = []
        self.env.player = random.choice([-1,1]) # randomize initial player
        self.env.reset()
        state = copy(self.env.board)
        while True:
            """ The Agent always acts as it would be player 1 (using illusion mechanism). """
            if self.env.legal_actions():
                action, q_value = self.agent.act(state, self.env.legal_actions(), self.env.player)
                next_observation, reward, done, info = self.env.step(action)
                if self.env.player == 1:
                    trajectory.append([state, action, q_value, 0, False])
                if done:
                    trajectory.append([state, 0, 0, reward, True])
                    break
            else:
                """ In case of a draw. """
                trajectory.append([state, 0, 0, 0, True])
                break
        return trajectory


    def run(self, n_iterations, episodes_in_batch, data_size, epochs):
        """Full RL training loop."""

        for num in range(n_iterations):
            """ Print current time."""
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {num+1}')

            """ Run batch of episodes. """
            for _ in tqdm(range(episodes_in_batch)):
                trajectory = self.run_one_episode(num)
                self.buffer.add_trajectory(trajectory, self.algorithm)

            print(f'Length of data in the buffer = {len(self.buffer.data)}')
            x, y = self.buffer.prepare_training_data()

            """ Fit the network to the data. """
            self.network.model.fit(x, y)
            print('Fitting finished')
            """ Clearing the buffer from the old trajectories.self.buffer.clear_buffer() """
            # TODO: to jest coś dziwnego
            # data_to_clear = episodes_in_batch * 3
            self.buffer.clear_buffer()
            """ Run test against heuristics. """
            self.tester.test_against_heuristic(self.network)

        print(f'Training finished.')




