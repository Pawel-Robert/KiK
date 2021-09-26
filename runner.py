""" Class of a runner. """

from tqdm import tqdm
from experience_buffer import ExperienceBuffer
from numpy import copy, random
from datetime import datetime
from tester import Tester


class Runner:
    def __init__(self, agent_class, algorithm_class, network, policy, env, epsilon=0.1, buffer_size=10000):
        """The main class to run the training loop"""
        self.network = network
        self.policy = policy
        self.agent = agent_class(network, policy, epsilon)
        self.env = env
        self.buffer = ExperienceBuffer(buffer_size)
        self.tester = Tester(env)
        self.algorithm = algorithm_class()

    def run_one_episode(self, iteration):
        """Plays one game AGAINST ITSELF and returns a trajectory of one player.
         In case of a draw, the loops ans and we store the reward.
         Remark: function step changes the player. We store the trajectory of player 1 only.
         The purpose of the policy trajectory is to gather training data for the policy network. """
        del iteration
        ai_player = 1 # agent always acts as it would be player 1 (using illusion mechanism)
        swap_const = -1
        trajectory = []
        # policy_trajectory = []
        self.env.player = random.choice([-1,1]) # randomize initial player
        self.env.reset()
        while True:
            if self.env.legal_actions():
                state = copy(self.env.board)
                action, q_value = self.agent.act(state * self.env.player, self.env.legal_actions())
                next_observation, reward, done, _ = self.env.step(action)
                if self.env.player * swap_const == ai_player:
                    trajectory.append([state, action, q_value, 0, False])
                    # policy_trajectory.append([state, sample_actions, q_values])
                if done:
                    trajectory.append([copy(next_observation), action, q_value, reward, True])
                    break
            else:
                trajectory.append([state, 0, 0, 0, True])
                break
        return trajectory#, policy_trajectory


    def run(self, n_iterations, episodes_in_batch, data_size, epochs):
        """Full RL training loop."""
        del data_size
        del epochs
        for num in range(n_iterations):
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {num+1}')

            for _ in tqdm(range(episodes_in_batch)):
                trajectory = self.run_one_episode(num)
                self.buffer.add_trajectory(trajectory, self.algorithm, self.network)
                # self.buffer.add_policy_trajectory(policy_trajectory)
            print(f'Length of data in the buffer = {len(self.buffer.data)}')

            states, q_values = self.buffer.prepare_training_data()
            self.network.model.fit(states, q_values)
            # states, distributions = self.buffer.prepare_policy_data()
            # self.policy.model.fit(states, distributions)
            print('Fitting finished')

            self.buffer.clear_buffer()
            # 2self.tester.test_against_random(self.network)

        print(f'Training finished.')
