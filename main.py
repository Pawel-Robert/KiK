#from KiK.value_network import ValueNetwork
from kik_env import KiKEnv
from q_network import QNetwork3x3
import numpy as np
from runner import Runner
from agent import Small_Agent_Explorator, Small_Agent


width = 3
height = 3
winning_condition = 3

network = QNetwork3x3()

env = KiKEnv(width, height, winning_condition)
#    Runner(agent_class, network, epsilon env, buffer_size, time_limit):
runner = Runner(Small_Agent_Explorator, network, 0.1, env, 1, 10)

#  run(n_iterations, episodes_in_batch, data_size, epochs)
# n_iterations = number of training processes
# episodes_in_batch = number of games played for one training
runner.run(1, 10, 100, 1)

env.reset()

#agent = Small_Agent(network)
#while True:
#    env.human_vs_ai_play(agent)

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
