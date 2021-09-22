""" Tester. """

from numpy import random, sqrt
from agent import Agent

class Tester:
    """ Tester. """
    def __init__(self, env):
        self.env = env

    def test_against_random(self, network):
        """ Test against heuristics. """
        network_wins = 0
        heuristic_wins = 0
        agent = Agent(network)
        for i in range(10):
            network_score, heuristic_score = self.run_one_episode_against_random(agent, i)
            network_wins += network_score
            heuristic_wins += heuristic_score
        print('\n TESTING PERCENTAGES:')
        print(f'Network wins: {network_wins}0 %')
        print(f'Heuristic wins: {heuristic_wins}0 %')
        draws = 100 - 10*network_wins - 10*heuristic_wins
        print(f'Draws: {draws} %')

    def compute_error(self, q_values, reward):
        error = sum([(q_values[i + 1] - q_values[i]) ** 2 for i in range(len(q_values) - 1)])
        return sqrt(error + (reward - q_values[-1])**2)

    def run_one_episode_against_random(self, agent, iteration):
        """Plays one game against HEURISTIC PLAYER and returns trajectory"""
        self.env.player = random.choice([-1, 1])
        self.env.reset()
        q_values = []
        while True:
           if self.env.legal_actions():
                if self.env.player == 1:
                    action, q_value = agent.act(self.env.board, self.env.legal_actions(), 1)
                    q_values.append(float("{:.2f}".format(float(q_value))))
                else:
                    action, _ = random.choice(self.env.legal_actions()), 0
           else:
               break
           _, reward, done, _ = self.env.step(action)
           if done:
               break
        if iteration < 10:
            print(q_values, reward)
        return max(reward, 0), -min(reward, 0)

