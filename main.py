#from KiK.value_network import ValueNetwork
from kik_env import KiKEnv
from q_network import QNetwork3x3
import numpy as np
from runner import Runner
from agent import Small_Agent_Explorator, Small_Agent
from keras.models import load_model
import time
import keras

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # Preprocess the data (these are NumPy arrays)
# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255
#
# y_train = y_train.astype("float32")
# y_test = y_test.astype("float32")
#
# # Reserve 10,000 samples for validation
# x_val = x_train[-10000:]
# y_val = y_train[-10000:]
# x_train = x_train[:-10000]
# y_train = y_train[:-100]
#
# print(x_train, y_train)
#
# time.sleep(100)

# INICJALIZACJA

width = 3
height = 3
winning_condition = 3

network = QNetwork3x3()

env = KiKEnv(width, height, winning_condition)
# env.human_vs_human_play()

# TRENING

# print('Podaj ilość iteracji w trakcie treningu przeciw heurystyce.')
# iterations = input()
# iterations = int(iterations)
# print('Podaj ilość rozgrywek w każdej iteracji przeciw heurystyce.')
# episodes = input()
# episodes = int(episodes)
# # print('Czy chcesz się połączyć z Neptunem?')
# # nept = False
#
#
# #    Runner(agent_class, network, epsilon, env, buffer_size, time_limit, heuristic):
# runner_1 = Runner(Small_Agent_Explorator, network, 0.1, env, 1, 10, True)
# runner_1.run(iterations, episodes, 100, 1, None)

print('Podaj ilość iteracji w trakcie treningu przeciw sobie.')
iterations = input()
iterations = int(iterations)
print('Podaj ilość rozgrywek w każdej iteracji przeciw sobie.')
episodes = input()
episodes = int(episodes)

runner_2 = Runner(Small_Agent_Explorator, network, 0.1, env, 1, 10, False)
runner_2.run(iterations, episodes, 100, 1, None)

#run(n_iterations, episodes_in_batch, data_size, epochs)
# n_iterations = number of training processes
# episodes_in_batch = number of games played for one training

# NEPTUN

#nept = True
# if nept:
#     import neptune.new as neptune
#     from neptune.new.integrations.tensorflow_keras import NeptuneCallback
#     run = neptune.init(project='pawel-robert/KiK',
#                        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOTY4NmVmNi02ODU1LTRkOGEtYjhmZS03MzlhYmJlNzM4YzYifQ==')  # add your credentials
#     neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
#     runner.run(iterations, episodes, 100, 1, neptune_cbk)
# else:
#     runner.run(iterations, episodes, 100, 1, None)

# SAVING MODEL
network.model.save("model_random_improved_Bellmans.h5")
# network.model = load_model('model_random_starting_player.h5')


# TESTING MODEL

env.reset()
#env.human_vs_human_play()
agent = Small_Agent(network)
while True:
   env.human_vs_ai_play(agent, network)

""" JAK UŻYĆ SIECI 3X3 """
# st = np.zeros(9)
# st[0] = 1.
# st_input = np.array([st])
# ac = np.zeros(9)
# ac[2] = -1.
# ac_input = np.array([ac])
# d = QNetwork3x3()
# print(d.model.predict([st_input, ac_input]))





#ValueNetwork(3, 3, None)

# trajectory = env.random_play()

# print(trajectory)

#env.human_vs_human_play()
