from experience_buffer import ExperienceBuffer
import neptune
from metrics import batch_metrics
from numpy import copy, random
from datetime import datetime
from agent import Heuristic_Agent
from tensorflow.keras.models import clone_model
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
# from joblib.externals.loky import set_loky_p
# import dill as pickle
# import ray

#import multiprocessing as mp

# ray.init()
# ray.available_resources()['CPU']
# set_loky_pickler("dill")

class Runner:
    def __init__(self, agent_class, network, epsilon, env, buffer_size, time_limit, heuristic, heigth, width, target_network_update_frequency=20):
        """The main class to run the training loop"""
        self.network = network
        self.agent = agent_class(network, epsilon)
        self.env = env
        self.time_limit = time_limit
        self.N = 0
        self.illegal_actions = 0
        """ Initialise experience buffer with a target network. """
        self.buffer = ExperienceBuffer(buffer_size, heigth, width, clone_model(network.model))
        self.heuristic = Heuristic_Agent()
        self.network_wins = 0
        self.heuristic_wins = 0
        self.heigth = heigth
        self.width = width
        self.target_network_update_frequency = target_network_update_frequency


    def run_one_episode(self, iteration):
        """Plays one game AGAINST ITSELF and returns two trajectories."""

        """Trajectory of the first player."""
        """ Each entry in the trajectory consists of states of both players, chosen action, 
         its Q-value, reward and the information, wheather the game has ended."""

        trajectory = []

        """Time counter."""
        t = 0

        """Randomise initial player."""
        self.env.player = random.choice([-1,1])
        # print(f'Zaczyna gracz {self.env.player}')
        observation = copy(self.env.reset())

        """ Main game loop. """
        while True:
            """Agent choose action from list of legal actions, if this list is nonempty."""
            """ The Agent always acts as it would be player 1. """
            """ Hence in the case, when agent shoudl act as player = - 1 we make an ''illusion'' 
            - we change the signs of the marks on the board, so that agent acts still as player 1. """

            state_1 = self.env.player_board(1)
            state_2 = self.env.player_board(-1)

            """ It the list is empty we have a draw. """
            if self.env.legal_actions():
                action, q_value = self.agent.act(state_1, state_2, self.env.legal_actions(), iteration, 1)
            else:
                """ In case of a draw. """
                trajectory.append([copy(state_1), copy(state_2), 0, 0, 0, True])
                break

            """ Punish the agent if it wants to make illegal move. In such case opponent wins. """
            # if action not in self.env.legal_actions():
            #    reward = -copy(self.env.player)
            #    trajectory_1.append([observation, 0, 0, reward])
            #    trajectory_2.append([observation, 0, 0, reward])
            #    break

            """Interact with the environment."""
            next_observation, reward, done, info = self.env.step(action)
            # if reward == 1:
            #     self.network_wins += 1
            # elif reward == -1:
            #     self.heuristic_wins += 1

            """ Add data to the trajectory (depending on the player)."""
            if self.env.player == 1:
                try:
                    trajectory.append([copy(state_1), copy(state_2), action, q_value, 0, False])
                except:
                    print("Error is appending trajectory.")
            # else:
            #     try:
            #         trajectory_2.append([copy(observation), action, q_value, 0])
            #     except:
            #         print("Error is appending trajectory.")
            if done:
                """ Add additional piece of trajectory, which is necessary for the Bellmans equation. """
                """ Only here we store the information about the reward. """
                # print(f'wygrał gracz {-self.env.player}, reward = {reward}')
                trajectory.append([copy(self.env.player_board(1)), copy(self.env.player_board(2)), 0, 0, reward, True])
                # trajectory_2.append([copy(next_observation), 0, 0, reward])
                break

            """ Update the observation. """
            observation = next_observation

            """Check time limit."""
            t += 1
            if t == self.time_limit:
                break

        return trajectory



    def run_batch_of_episodes(self, n_episodes, iteration):
        """Runs a batch of episeodes and returns list of trajectories."""
        """This could be parallelized to run on many cpus"""
        trajectory_batch = []

        # trajectory_batch = Parallel(n_jobs=6, prefer="threads")(delayed(Runner.run_one_episode)(self) for _ in range(n_episodes))

        for i in range(n_episodes):
            trajectory_batch.append(self.run_one_episode(iteration))

        return trajectory_batch


    def pre_training(self, n_iterations, episodes_in_batch, data_size, epochs, callback):
        """ Initial training with more episodes in an iteration."""
        for num in range(n_iterations):
            """ Print current time."""
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print('Pretraining.')
            """ Run batch of episodes. """
            trajectory_batch = self.run_batch_of_episodes(episodes_in_batch, num)
            """ Add data to the memory buffer and prepare data for training. """
            for trajectory in trajectory_batch:
                self.buffer.add_trajectory(trajectory)
            x, y = self.buffer.prepare_training_data(data_size)
            """ Fit the network to the data. """
            self.network.model.fit(x, y)

        print(f'Training finished.')

    def run(self, n_iterations, episodes_in_batch, data_size, epochs):
        """Full RL training loop."""
        """ statistical data for benchmarks """
        network_wins_list = []
        heuristic_wins_list = []
        draws_list = []
        for num in range(n_iterations):

            """ Print current time."""
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {num+1}')

            """ Run batch of episodes. """
            trajectory_batch = self.run_batch_of_episodes(episodes_in_batch, num)


            """ Print various metrics. """
            # print(f'First win rate = {batch_metrics(trajectory_batch, num)[0]}')
            # print(f'Second win rate = {batch_metrics(trajectory_batch, num)[1]}')
            # print(f'Draws = {batch_metrics(trajectory_batch, num)[2]}')
            # print(f'Average game length = {batch_metrics(trajectory_batch, num)[3]}')

            """ Add data to the memory buffer and prepare data for training. """
            self.buffer.lengths_of_trajectories = 0
            for trajectory in trajectory_batch:
                self.buffer.add_trajectory(trajectory)
            x, y = self.buffer.prepare_training_data(data_size)


            """ Fit the network to the data. """
            self.network.model.fit(x, y)

            """ Once for a while update the target network. """
            if num % self.target_network_update_frequency == 0:
                self.buffer.target_network_model = clone_model(self.network.model)
            #
            # """ Update target network in experience buffer once a while"""
            # if
            # self.buffer.target_network = self.network

            """ In case of Neptune. """
            # self.network.model.fit(x, y, epochs, callbacks=[callback])


            """ Clearing the buffer from the old trajectories.self.buffer.clear_buffer() """
            self.buffer.clear_buffer()

            """ Test against heuristics. """
            self.network_wins = 0
            self.heuristic_wins = 0
            for _ in range(100):
                self.run_one_episode_against_heuristic(10000)
            print('TESTING PERCENTAGES:')
            print(f'Network wins = {self.network_wins}')
            print(f'Heuristic wins = {self.heuristic_wins}')
            draws = 100 - self.network_wins - self.heuristic_wins
            print(f'Draws = {draws}')
            network_wins_list.append(self.network_wins)
            heuristic_wins_list.append(self.heuristic_wins)
            draws_list.append(draws)
            # neptune.log_metric('network_wins', network_wins)

        print(f'Training finished.')


    def run_one_episode_against_heuristic(self, iteration):
        """Plays one game against HEURISTIC PLAYER and returns trajectory"""

        """Randomise initial player."""
        self.env.player = random.choice([-1, 1])
        observation = copy(self.env.reset())

        while True:
           if self.env.legal_actions():
                if self.env.player == 1:
                    action, q_value = self.agent.act(self.env.player_board(1), self.env.player_board(-1), self.env.legal_actions(), iteration, 1)
                else:
                    action, q_value = self.heuristic.act(observation, self.env.legal_actions(), -1)
           else:
               break

           next_observation, reward, done, info = self.env.step(action)

           if reward == 1:
                self.network_wins += 1
           elif reward == -1:
                self.heuristic_wins += 1

           if done:
                break

           observation = next_observation

        return reward

