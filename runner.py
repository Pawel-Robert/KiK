import time

from tqdm import tqdm

from experience_buffer import ExperienceBuffer
from numpy import copy, random
from datetime import datetime
from agent import Heuristic_Agent, Random_Agent
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
        self.heuristic = Random_Agent()
        self.network_wins = 0
        self.heuristic_wins = 0
        self.heigth = heigth
        self.width = width
        self.target_network_update_frequency = target_network_update_frequency


    def run_one_episode(self, iteration):
        time_s = time.time()
        """Plays one game AGAINST ITSELF and returns a trajectory of one player."""

        """Trajectory of the first player."""
        """ Each entry in the trajectory consists of states of both players, chosen action, 
         its Q-value, reward and the information, wheather the game has ended."""

        trajectory = []

        """Time counter."""
        t = 0

        """Randomise initial player."""
        self.env.player = random.choice([-1,1])
        self.env.reset()
        """ Main game loop. """
        while True:
            """Agent choose action from list of legal actions, if this list is nonempty."""
            """ The Agent always acts as it would be player 1. """
            """ Hence in the case, when agent should act as player = - 1 we make an ''illusion'' 
            - we change the signs of the marks on the board, so that agent acts still as player 1. """

            state = self.env.board

            """ It the list of legal actions is empty we have a draw. """
            if self.env.legal_actions():
                action, q_value = self.agent.act(state, self.env.legal_actions(), iteration, 1)
                """Interact with the environment."""
                next_observation, reward, done, info = self.env.step(action)
                if self.env.player == 1:
                    try:
                        trajectory.append([copy(state), action, q_value, 0, False])
                    except:
                        print("Error is appending trajectory.")
                if done:
                    """ Add additional piece of trajectory, which is necessary for the Bellmans equation. """
                    """ Only here we store the information about the reward. """
                    # We store this information regardless who made this last move.
                    #
                    trajectory.append([copy(self.env.board), 0, 0, reward, True])
                    break
            else:
                """ In case of a draw. """
                trajectory.append([copy(state), 0, 0, 0, True])
                break

            """Check time limit."""
            t += 1
            if t == self.time_limit:
                break

        print(f'Playing game took = {time.time() - time_s}')
        return trajectory



    def run_batch_of_episodes(self, n_episodes, iteration):
        """Runs a batch of episeodes and returns list of trajectories."""
        """This could be parallelized to run on many cpus"""
        trajectory_batch = []

        # trajectory_batch = Parallel(n_jobs=6, prefer="threads")(delayed(Runner.run_one_episode)(self) for _ in range(n_episodes))

        for i in tqdm(range(n_episodes)):
            trajectory_batch.append(self.run_one_episode(iteration))
            # print('Run one episode.')

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
            #    print('Add trajectory!')
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

            """ Add data to the memory buffer and prepare data for training. """
            # self.buffer.lengths_of_trajectories = 0
            for trajectory in trajectory_batch:
                self.buffer.add_trajectory(trajectory)
            print(f'Length of data in the buffer = {len(self.buffer.data)}')
            print(f'Total lenght of trajectories = {self.buffer.lengths_of_trajectories}')
            x, y = self.buffer.prepare_training_data(data_size)


            """ Fit the network to the data. """
            self.network.model.fit(x, y)

            """ Clearing the buffer from the old trajectories.self.buffer.clear_buffer() """
            # TODO: to jest coś dziwnego
            # data_to_clear = episodes_in_batch * 3
            self.buffer.clear_buffer(0)

            """ Run test against heuristics. """
            network_wins_list, heuristic_wins_list, draws_list = self.test_against_heuristic(network_wins_list, heuristic_wins_list, draws_list)

        print(f'Training finished.')

    def test_against_heuristic(self, network_wins_list, heuristic_wins_list, draws_list):
        """ Test against heuristics. """
        self.network_wins = 0
        self.heuristic_wins = 0
        for _ in range(100):
            self.run_one_episode_against_heuristic(10000)
        print('TESTING PERCENTAGES:')
        print(f'Network wins = {self.network_wins} %')
        print(f'Heuristic wins = {self.heuristic_wins} %')
        draws = 100 - self.network_wins - self.heuristic_wins
        print(f'Draws = {draws}')
        network_wins_list.append(self.network_wins)
        heuristic_wins_list.append(self.heuristic_wins)
        draws_list.append(draws)
        return network_wins_list, heuristic_wins_list, draws_list

    def run_one_episode_against_heuristic(self, iteration):
        """Plays one game against HEURISTIC PLAYER and returns trajectory"""

        """Randomise initial player."""
        self.env.player = random.choice([-1, 1])
        observation = copy(self.env.reset())

        while True:
           if self.env.legal_actions():
                if self.env.player == 1:
                    action, q_value = self.agent.act(self.env.board, self.env.legal_actions(), iteration, 1)
                else:
                    x = self.env.legal_actions()
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

