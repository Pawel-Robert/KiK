from experience_buffer import ExperienceBuffer
from metrics import batch_metrics
from numpy import copy, random
from datetime import datetime
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler
# import dill as pickle
# import ray

#import multiprocessing as mp

# ray.init()
# ray.available_resources()['CPU']
set_loky_pickler("dill")

class Runner:
    def __init__(self, agent_class, network, epsilon, env, buffer_size, time_limit, heuristic):
        """The main class to run the training loop"""
        # tworzymy agenta o typie podanym w parametrze agent_class
        self.network = network
        self.agent = agent_class(network, epsilon)
        self.env = env
        self.time_limit = time_limit
        self.N = 0
        self.illegal_actions = 0
        self.buffer = ExperienceBuffer(buffer_size)
        self.heuristic = heuristic


    def run_one_episode(self):
        """Plays one game and returns trajectory"""
        trajectory_1 = []
        trajectory_2 = []
        t = 0
        #self.N += 1
        self.env.player = 2*random.randint(2)-1
        #print(f'zaczyna gracz {self.env.player}')
        observation = copy(self.env.reset())
        #print(observation)
        while True:
            # self.env.render()
            # agent wybiera akcję na polu jeszcze nie zajętym
            if self.env.legal_actions():
                action, q_value = self.agent.act(observation, self.env.legal_actions(), self.N, self.env.player)
            else:
                # print(f'draw, reward = {reward}')
                break
            #if action in self.env.legal_actions():
            next_observation, reward, done, info = self.env.step(action)
            # kara dla agenta, który chciałby wykonać nielegalny ruch
            # wygrywa wówczas gracz przeciwny
            # else:
            #    reward = -self.env.player
            #    done = True
            #    self.illegal_actions += 1
            # print(observation)
            # wybraną akcją wpływamy na środowiski i zbieramy obserwacje

            # print(observation, next_observation)
            # obserwacje dodajemy do trajektorii
            # q_value = q_value/1.01
            """ We have two trajectories for two different players. Using each trajectory and Bellmans equation we will train our network."""
            if self.env.player == 1:
                try:
                    trajectory_1.append([copy(observation), copy(action), copy(q_value), copy(reward)])
                except:
                    print("Error is appending trajectory.")
                if done:
                #print(f'done, reward={reward}, player = {self.env.player}')
                # musimy do trajektorii dodać jeszcze stan ostatni planszy
                    trajectory_1.append([copy(next_observation), 0, 0, 0, 0])
                    break
            else:
                try:
                    trajectory_2.append([copy(observation), copy(action), copy(q_value), copy(reward)])
                except:
                    print("Error is appending trajectory.")
                if done:
                #print(f'done, reward={reward}, player = {self.env.player}')
                # musimy do trajektorii dodać jeszcze stan ostatni planszy
                    trajectory_2.append([copy(next_observation), 0, 0, 0, 0])
                    break


            # update our observatons
            observation = next_observation
            # # sprawdzmy, czy nie przekroczyliśmy limitu czasu
            t += 1
            if t == self.time_limit:
                break
        # self.env.render()
        # print(done)
        return trajectory_1, trajectory_2

    def run_one_episode_against_heuristic(self):
        """Plays one game and returns trajectory"""
        trajectory = []
        t = 0
        #self.N += 1
        self.env.player = 2*random.randint(2)-1
        heuristic_player = 2*random.randint(2)-1
        #print(f'zaczyna gracz {self.env.player}')
        observation = copy(self.env.reset())
        #print(observation)
        while True:
            if self.env.player == heuristic_player:
                next_observation, reward, done, info = self.env.heuristic()
            else:

                # self.env.render()
                # agent wybiera akcję na polu jeszcze nie zajętym
                if self.env.legal_actions():
                    action, q_value = self.agent.act(observation, self.env.legal_actions(), self.N, self.env.player)
                else:
                    # print(f'draw, reward = {reward}')
                    break
                if action in self.env.legal_actions():
                    next_observation, reward, done, info = self.env.step(action)
                # kara dla agenta, który chciałby wykonać nielegalny ruch
                # wygrywa wówczas gracz przeciwny
                else:
                    reward = -self.env.player
                    done = True
                    self.illegal_actions += 1
                # rint(observation)
                # wybraną akcją wpływamy na środowiski i zbieramy obserwacje

                # print(observation, next_observation)
                # obserwacje dodajemy do trajektorii
                # q_value = q_value/1.01
            try:
                trajectory.append([copy(observation), copy(action), copy(q_value), copy(reward)])
            except:
                print("Error in appending trajectory.")
            if done:
                #print(f'done, reward={reward}, player = {self.env.player}')
                # musimy do trajektorii dodać jeszcze stan ostatni planszy
                trajectory.append([copy(next_observation), 0, 0, 0, 0])
                break

            # update our observatons
            observation = next_observation
            # # sprawdzmy, czy nie przekroczyliśmy limitu czasu
            t += 1
            if t == self.time_limit:
                break
        # self.env.render()
        # print(done)
        return trajectory

    def run_batch_of_episodes(self, n_episodes):
        """This could be parallelized to run on many cpus"""
        trajectory_batch = []
        # trajectory_batch = Parallel(n_jobs=6, prefer="threads")(delayed(Runner.run_one_episode)(self) for _ in range(n_episodes))
        for i in range(n_episodes):
            temp_list = self.run_one_episode()
            # print(temp_list[0])
            # print(temp_list[1])
            trajectory_batch.append(temp_list[0])
            trajectory_batch.append(temp_list[1])
        # print(trajectory_batch)
        return trajectory_batch

    def run(self, n_iterations, episodes_in_batch, data_size, epochs, callback):
        """Full RL training loop"""
        for num in range(n_iterations):
            self.buffer.clear_buffer()
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {num+1}')
            trajectory_batch = self.run_batch_of_episodes(episodes_in_batch)
            # print(f'First win rate = {batch_metrics(trajectory_batch, num)[0]}')
            # print(f'Second win rate = {batch_metrics(trajectory_batch, num)[1]}')
            # print(f'Draws = {batch_metrics(trajectory_batch, num)[2]}')
            # print(f'Average game length = {batch_metrics(trajectory_batch, num)[3]}')
            # print(f'Number of illegal actions = {self.illegal_actions}')
            self.illegal_actions = 0
            for trajectory in trajectory_batch:
                self.buffer.add_trajectory(trajectory)
            x, y = self.buffer.prepare_training_data(data_size)
            # print(x,y)
            if callback is None:
                train_metrics = self.network.model.fit(x, y)
            else:
                train_metrics = self.network.model.fit(x, y, epochs, callbacks=[callback])


        print(f'Training finished.')