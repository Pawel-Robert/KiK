#from KiK.value_network import ValueNetwork
from kik_env import KiKEnv
from q_network import QNetwork3x3
import numpy as np
from runner import Runner
from agent import Small_Agent_Explorator, Small_Agent
from keras.models import load_model

print('Podaj ilość iteracji w trakcie treningu.')
iterations = input()
iterations = int(iterations)
print('Podaj ilość rozgrywek w każdej iteracji.')
episodes = input()
episodes = int(episodes)
# print('Czy chcesz się połączyć z Neptunem?')
nept = False

width = 3
height = 3
winning_condition = 3

network = QNetwork3x3()

env = KiKEnv(width, height, winning_condition)
# env.human_vs_human_play()
#    Runner(agent_class, network, epsilon env, buffer_size, time_limit):
runner = Runner(Small_Agent_Explorator, network, 0.1, env, 1, 10)

#  run(n_iterations, episodes_in_batch, data_size, epochs)
# n_iterations = number of training processes
# episodes_in_batch = number of games played for one training

#nept = True
if nept:
    import neptune.new as neptune
    from neptune.new.integrations.tensorflow_keras import NeptuneCallback
    run = neptune.init(project='pawel-robert/KiK',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOTY4NmVmNi02ODU1LTRkOGEtYjhmZS03MzlhYmJlNzM4YzYifQ==')  # add your credentials
    neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')
    runner.run(iterations, episodes, 100, 1, neptune_cbk)
else:
    runner.run(iterations, episodes, 100, 1, None)

# save the result of training
network.model.save("model_i=100_g=500.h5")
#network.model = load_model('model_i=10_g=100.h5')

#env.reset()
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
