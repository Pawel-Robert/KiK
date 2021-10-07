""" Tester. """

from numpy import random
from agent import Agent
from numpy import copy
from tqdm import tqdm
from networks import ValueNetwork

class Arena:
    """ Tester. """
    def __init__(self, env, number_of_duels):
        self.env = env
        self.number_of_duels = number_of_duels

    def many_duels(self, number, network_1, network_2):
        wins_1 = 0
        wins_2 = 0
        for _ in tqdm(range(number)):
            win_1, win_2 = self.duel(network_1, network_2)
            wins_1 += win_1
            wins_2 += win_2
        return wins_1, wins_2

    def duel(self, network_1, network_2):
        """ Duel of two networks. """
        self.env.player = random.choice([-1, 1])
        self.env.reset()
        agent_1 = Agent(network_1)
        agent_2 = Agent(network_2)
        while True:
            if self.env.legal_actions():
                state = copy(self.env.board)
                if self.env.player == 1:
                    action, q_value = agent_1.act(state * self.env.player, self.env.legal_actions())
                else:
                    action, q_value = agent_2.act(state * self.env.player, self.env.legal_actions())
            else:
                break
            _, reward, done, _ = self.env.step(action)
            if done:
                break
        return max(reward, 0), -min(reward, 0)

    def tournament(self, past_networks, network):
        """ Games between various networks. """
        i = 1
        wins = []
        loses = []
        for past_network in past_networks:
            print("\n Dueling.")
            wins_new, wins_old = self.many_duels(self.number_of_duels, network, past_network)
            wins.append(wins_new)
            loses.append(wins_old)
            print(f'\n Wins against network nr {i} = {wins_new}')
            i += 1
        temp_network = ValueNetwork(self.env.height, self.env.width)
        for a, b in zip(temp_network.model.variables, network.model.variables):
            a.assign(b)
        return temp_network, wins, loses
