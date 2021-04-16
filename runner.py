from experience_buffer import ExperienceBuffer
from metrics import batch_metrics
from numpy import copy
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
    def __init__(self, agent_class, network, epsilon, env, buffer_size, time_limit):
        """The main class to run the training loop"""
        # tworzymy agenta o typie podanym w parametrze agent_class
        self.network = network
        self.agent = agent_class(network, epsilon)
        self.env = env
        self.time_limit = time_limit
        self.N = 0

        self.buffer = ExperienceBuffer(buffer_size)

    # @ray.remote
    @delayed
    @wrap_non_picklable_objects
    def run_one_episode(self):
        """Plays one game and returns trajectory"""
        trajectory = []
        t = 0
        self.N += 1

        observation = copy(self.env.reset())
        #print(observation)
        while True:
            # agent wybiera akcję na polu jeszcze nie zajętym
            if self.env.legal_actions():
                action, q_value = self.agent.act(observation, self.env.legal_actions(), self.N, self.env.player)
            else:
                #print(f'draw, reward = {reward}')
                break
            #rint(observation)
            # wybraną akcją wpływamy na środowiski i zbieramy obserwacje
            next_observation, reward, done, info = self.env.step(action)
            # print(observation, next_observation)
            # obserwacje dodajemy do trajektorii
            # q_value = q_value/1.01
            try:
                trajectory.append([copy(observation), copy(action), copy(q_value), copy(reward)])
            except:
                print(q_value)
            if done:
                #print(f'done, reward={reward}')
                # musimy do trajektorii dodać jeszcze stan ostatni planszy
                trajectory.append([copy(next_observation), 0, 0, 0, 0])
                break



            # update our observatons
            observation = next_observation
            # sprawdzmy, czy nie przekroczyliśmy limitu czasu
            t += 1
            if t == self.time_limit:
                break
        # self.env.render()
        # print(done)
        return trajectory

    def run_batch_of_episodes(self, n_episodes):
        """This could be parallelized to run on many cpus"""
        trajectory_batch = []
        trajectory_batch = Parallel(n_jobs=4)(self.run_one_episode() for _ in range(n_episodes))
        print(trajectory_batch)
        return trajectory_batch

    def run(self, n_iterations, episodes_in_batch, data_size, epochs, callback):
        """Full RL training loop"""
        for num in range(n_iterations):
            self.buffer.clear_buffer()
            now = datetime.now().time()
            print(now.strftime("%H:%M:%S"))
            print(f'Processing step = {num+1}')
            trajectory_batch = self.run_batch_of_episodes(episodes_in_batch)
            print(f'First win rate = {batch_metrics(trajectory_batch, num)[0]}')
            print(f'Second win rate = {batch_metrics(trajectory_batch, num)[1]}')
            print(f'Draws = {batch_metrics(trajectory_batch, num)[2]}')
            print(f'Average game length = {batch_metrics(trajectory_batch, num)[3]}')
            for trajectory in trajectory_batch:
                self.buffer.add_trajectory(trajectory)
            x, y = self.buffer.prepare_training_data(data_size)
            if callback == None:
                train_metrics = self.network.model.fit(x, y)
            else:
                train_metrics = self.network.model.fit(x, y, epochs, callbacks=[callback])


        print(f'Training finished.')







        