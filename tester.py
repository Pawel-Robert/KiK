from numpy import random, copy
from agent import Small_Agent
from kik_env import KiKEnv

class Tester:
    def __init__(self, env):
        self.env = env

    def test_against_random(self, network):
        """ Test against heuristics. """
        network_wins = 0
        heuristic_wins = 0
        agent = Small_Agent(network)
        for _ in range(100):
            a, b = self.run_one_episode_against_random(agent, 1000)
            network_wins += a
            heuristic_wins += b
        print('\n TESTING PERCENTAGES:')
        print(f'Network wins: {network_wins} %')
        print(f'Heuristic wins: {heuristic_wins} %')
        draws = 100 - network_wins - heuristic_wins
        print(f'Draws: {draws} %')

    def run_one_episode_against_random(self, agent, iteration):
        """Plays one game against HEURISTIC PLAYER and returns trajectory"""
        del iteration
        self.env.player = random.choice([-1, 1])
        while True:
           if self.env.legal_actions():
                if self.env.player == 1:
                    action, _ = agent.act(self.env.board, self.env.legal_actions(), 1)
                else:
                    action, _ = random.choice(self.env.legal_actions()), 0
           else:
               break
           _, reward, done, _ = self.env.step(action)
           if reward == 1:
                return 1, 0
           elif reward == -1:
                return 0, 1
           if done:
                break
        return 0, 0

