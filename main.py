""" Main file. """
from networks import ValueNetwork
from runner import Runner
from agent import AgentExplorator, Agent
from training_algorithms import BellmanAlgorithm
from datetime import datetime
from kik_env import KiKEnv
from player import Gomoku
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from callback import NeptuneLogger
import numpy as np
from tensorflow.keras.models import load_model

# Environment parameters.
WIDTH = 13
HEIGHT = 13
WIN_CND = 5

# Learning parameters.
EPSILON = 0.5
ALPHA = 0.8
GAMMA = 0.95
ITERATIONS = 500
EPISODES = 100

# Other parameters.
NUMBER_OF_DUELS = 100
HUMAN_TEST = True


run = neptune.init(project='pawel-robert/KiK',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOTY4NmVmNi02ODU1LTRkOGEtYjhmZS03MzlhYmJlNzM4YzYifQ==')  # add your credentials
# neptune.log_metrics('LOG_NAME', THING_I_WANT_TO_LOG)
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')


env = KiKEnv(WIDTH, HEIGHT, WIN_CND)
network = ValueNetwork(WIDTH, HEIGHT)
# network.model = load_model('./models_big/model_size_13_date_30.h5')
runner = Runner(AgentExplorator, BellmanAlgorithm(ALPHA, GAMMA), network, env, neptune_cbk, run,
                EPSILON, NUMBER_OF_DUELS, 100)
training_data_size, all_wins = runner.run(ITERATIONS, EPISODES)

print(f'Length of training data = {training_data_size}')
print(f'Wins over older networks = {all_wins}')
now = datetime.now().time()
network.model.save(f'./models_big/model_size_13_date_1_10.h5')
print(f'Network saved.')

if HUMAN_TEST:
    agent = Agent(network)
    env.game_play(agent)



