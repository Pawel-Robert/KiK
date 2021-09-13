""" Main file. """
from q_network import QValue
from runner import Runner
from agent import Small_Agent_Explorator, Small_Agent
from training_algorithms import BellmanAlgorithm
from datetime import datetime
from kik_env import KiKEnv
from tensorflow.keras.models import load_model

WIDTH = 3
HEIGHT = 3
WIN_CND = 3

env = KiKEnv(WIDTH, HEIGHT, WIN_CND)
network = QValue()
runner = Runner(Small_Agent_Explorator, BellmanAlgorithm, network, env, 0.1, 100)
# runner.run(100, 500, None, 1)
now = datetime.now().time()
#network.model.save(f'./models/model_{now.strftime("%H:%M:%S")}.h5')
network.model = load_model('./models/good_model.h5')
HUMAN_TEST = True
if HUMAN_TEST:
    agent = Small_Agent(network)
    env.game_play(agent, network)



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


