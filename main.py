from KiK.value_network import ValueNetwork
from kik_env import KiKEnv
import numpy as np

width = 3
height = 3
winning_condition = 3

env = KiKEnv(width, height, winning_condition)

ValueNetwork(3, 3, None)

# trajectory = env.random_play()

# print(trajectory)


# env.human_vs_human_play()
