""" Class of a runner. """

from tqdm import tqdm
from experience_buffer import ExperienceBuffer
from numpy import copy, random, sqrt
from datetime import datetime
from arena import Arena
from networks import ValueNetwork


class Runner:
    def __init__(self, agent_class, algorithm, network, env, callback, neptune_run, epsilon=0.1, number_of_duels = 100, buffer_size=10000):
        """The main class to run the training loop"""
        self.network = network
        self.agent = agent_class(network, epsilon)
        self.env = env
        self.callback = callback
        self.buffer = ExperienceBuffer(buffer_size)
        self.arena = Arena(env, number_of_duels)
        self.algorithm = algorithm
        self.training_data_size = 0
        self.past_networks = []
        self.neptune_run = neptune_run
        self.epsilon = epsilon

    def run_one_episode(self, iteration):
        """Plays one game AGAINST ITSELF and returns a trajectory of one player.
         In case of a draw, the loops ans and we store the reward.
         Remark: function step changes the player. We store the trajectory of player 1 only.
         The purpose of the policy trajectory is to gather training data for the policy network. """
        ai_player = 1 # agent always acts as it would be player 1 (using illusion mechanism)
        swap_const = -1
        trajectory = []
        self.env.player = random.choice([-1,1]) # randomize initial player
        self.env.reset()
        while True:
            if self.env.legal_actions():
                state = copy(self.env.board)
                action, q_value = self.agent.act(state * self.env.player, self.env.legal_actions(), iteration)
                next_observation, reward, done, _ = self.env.step(action)
                if self.env.player * swap_const == ai_player:
                    trajectory.append([state, action, q_value, 0, False])
                if done:
                    trajectory.append([copy(next_observation), action, q_value, reward, True])
                    break
            else:
                trajectory.append([state, 0, 0, 0, True])
                break
        return trajectory


    def run(self, n_iterations, episodes_in_batch):
        """Full RL training loop."""
        all_wins = []
        for iteration in range(n_iterations):
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {iteration+1}')

            for _ in tqdm(range(episodes_in_batch)):
                trajectory = self.run_one_episode(iteration)
                self.training_data_size += len(trajectory)
                self.buffer.add_trajectory(trajectory, self.algorithm, self.network)
            self.neptune_run['epsilon'].log(self.epsilon/sqrt(iteration + 1))
            print(f'Length of data in the buffer = {len(self.buffer.data)}')

            states, q_values = self.buffer.prepare_training_data()
            self.neptune_run['data_lenght'].log(states[0].shape[0])
            self.network.model.fit(states, q_values, callbacks=self.callback)
            print('Fitting finished')
            self.buffer.clear_buffer()
            if (iteration + 1) % 10 == 0:
                temp_network, wins, loses = self.arena.tournament(self.past_networks, self.network)
                self.past_networks = [temp_network]
                if len(wins) > 0:
                    self.neptune_run['against_recent_wins'].log(wins[-1])
                for win in wins:
                    all_wins.append(win)
                # if wins is not None:
                #     self.neptune_run["wins"].log(wins)

        print(f'Training finished.')
        return self.training_data_size, all_wins
