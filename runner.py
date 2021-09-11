""" Class of a runner. """

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
        self.buffer = ExperienceBuffer(buffer_size)
        self.tester = Tester(env)
        self.algorithm = algorithm_class()

    def run_one_episode(self, iteration):
        """Plays one game AGAINST ITSELF and returns a trajectory of one player.
         The Agent always acts as it would be player 1 (using illusion mechanism).
         In case of a draw, the loops ans and we store the reward.
         Remark: function step changes the player. We store the trajectory of player 1 only. """
        del iteration
        ai_player = 1
        swap_const = -1
        trajectory = []
        self.env.player = random.choice([-1,1]) # randomize initial player
        self.env.reset()
        while True:
            if self.env.legal_actions():
                state = copy(self.env.board)
                action, q_value = self.agent.act(state * self.env.player, self.env.legal_actions(), self.env.player)
                next_observation, reward, done, _ = self.env.step(action)
                if self.env.player * swap_const == ai_player:
                    trajectory.append([state, action, q_value, 0, False])
                if done:
                    trajectory.append([state, action, q_value, reward, True])
                    break
            else:
                trajectory.append([state, 0, 0, 0, True])
                break
        return trajectory


    def run(self, n_iterations, episodes_in_batch, data_size, epochs):
        """Full RL training loop."""
        del data_size
        del epochs
        target_network = self.network
        for num in range(n_iterations):
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {num+1}')

            for _ in tqdm(range(episodes_in_batch)):
                trajectory = self.run_one_episode(num)
                self.buffer.add_trajectory(trajectory, self.algorithm, target_network)
            print(f'Length of data in the buffer = {len(self.buffer.data)}')

            state_and_actions, q_values = self.buffer.prepare_training_data()
            self.network.model.fit(state_and_actions, q_values)
            print('Fitting finished')
            if num % 10 == 0:
                target_network = self.network

            self.buffer.clear_buffer()
            self.tester.test_against_random(self.network)

        print(f'Training finished.')
