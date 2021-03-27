from kik_env import KiKEnv
import numpy as np

width = 3
height = 3
winning_condition = 3

env = KiKEnv(width, height, winning_condition)

trajectory = env.random_play()

#print(trajectory)

env.human_vs_human_play()
