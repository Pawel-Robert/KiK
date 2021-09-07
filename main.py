#from KiK.value_network import ValueNetwork
from kik_env import KiKEnv
from q_network import QValue, ValueNetwork3x3
import numpy as np
from runner import Runner
from agent import Small_Agent_Explorator, Small_Agent, Heuristic_Agent, Random_Agent, Small_MCTS_Agent
from keras.models import load_model
import time
import keras
from MCTS import MonteCarloTreeSearch
import random

random.randint(0,1)




""" Defining environement. """""

width = 3
height = 3
winning_condition = 3

env = KiKEnv(width, height, winning_condition)


""" Defining the network. """

network = QValue()

""" Inicialising Neptune. """
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



""" Training the network. """

runner = Runner(Small_Agent_Explorator, network, 0.1, env, 100, 10, True, height, width)
print('Podaj ilość iteracji w trakcie treningu.')
# iterations = int(input())
print('Podaj ilość rozgrywek w każdej iteracji.')
# episodes = int(input())
print('Podaj ilość danych treningowych w każdym treningu.')
# data_size = int(input())
# runner.pre_training(1, 300, 100, 1, neptune_cbk)
# runner.run(iterations, episodes, 100, 1, neptune_cbk)
# runner.run(iterations, episodes, None, 1)

runner.run(100, 500, None, 1)

""" Saving and loading model trained."""


network.model.save("model_august_01.h5")
#network.model = load_model('model_random_maj_4_1.h5')

print('Zapisano model')

""" Testing the network through gameplay. """

human_test = False

if human_test:
    agent = Small_Agent(network)
    env.game_play(agent, network)

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


