from kik_env import KiKEnv
import numpy as np

width = 6
height = 6

env = KiKEnv(width, height)

# trajectory = env.random_play()

# print(trajectory)

env.human_vs_human_play()
