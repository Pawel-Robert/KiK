""" Main file. """
from networks import ValueNetwork, PolicyNetwork
from runner import Runner
from agent import AgentExplorator, Agent, PolicyAgentExplorator, PolicyAgent
from training_algorithms import BellmanAlgorithm
from datetime import datetime
from kik_env import KiKEnv
import numpy as np
from tensorflow.keras.models import load_model

WIDTH = 13
HEIGHT = 13
WIN_CND = 5
META_ITERATIONS = 1
ITERATIONS = 500
EPISODES = 100
HUMAN_TEST = True

env = KiKEnv(WIDTH, HEIGHT, WIN_CND)
network = ValueNetwork(WIDTH, HEIGHT)
network.model = load_model('./models_big/model_19:11:57.h5')
policy = PolicyNetwork(WIDTH, HEIGHT)
runner = Runner(AgentExplorator, BellmanAlgorithm, network, policy, env, 0.2, 100)
for _ in range(META_ITERATIONS):
    runner.run(ITERATIONS, EPISODES, None, 1)
    now = datetime.now().time()
    network.model.save(f'./models_big/model_{now.strftime("%H:%M:%S")}.h5')

if HUMAN_TEST:
    agent = Agent(network, policy)
    env.game_play(agent)



# """ Inicialising Neptune. """
#
# import neptune.new as neptune
# from neptune.new.integrations.tensorflow_keras import NeptuneCallback
# run = neptune.init(project='pawel-robert/KiK',
#                        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOTY4NmVmNi02ODU1LTRkOGEtYjhmZS03MzlhYmJlNzM4YzYifQ==')  # add your credentials
# neptune.log_metrics('LOG_NAME', THING_I_WANT_TO_LOG)

# class NeptuneLogger(NeptuneCallback):
#
#     def on_batch_end(self, batch, logs={}):
#         for log_name, log_value in logs.items():
#             neptune.log_metric(f'batch_{log_name}', log_value)
#
#     def on_epoch_end(self, epoch, logs={}):
#         for log_name, log_value in logs.items():
#             neptune.log_metric(f'epoch_{log_name}', log_value)

# neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')



# print('Podaj ilość iteracji w trakcie treningu.')
# # iterations = int(input())
# print('Podaj ilość rozgrywek w każdej iteracji.')
# # episodes = int(input())
# print('Podaj ilość danych treningowych w każdym treningu.')
# data_size = int(input())



#  TO JEST TAKI TEST NA MCTS
# value_network = ValueNetwork3x3()
# env.reset()
# mcts = MonteCarloTreeSearch(network, env)
#
# env.render()
# for _ in range(4):
#     action = mcts.predict_action(env.board)
#     env.step(action)
#     env.render()


