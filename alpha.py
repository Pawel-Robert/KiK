""" Monte Carlo Tree search. """

from node import Node
from numpy import random


class MonteCarloTreeSearch(Search, env, network):
    """ Monte Carlo Tree search algorithm. """
    def __init__(self):
        self.env = env
        self.network = network

    def select(node):
        """ Selects a leaf, which is a child of the node. """
        return random.choice(node.children)

    def expand(leaf):
        """ Expands a leaf, creating its children. """
        env.board = leaf.state
        action = random.choice(env.legal_actions())
        state, reward, done, _ = env.step(action)
        new_node = Node(leaf, state)
        leaf.add_child(new_node)
        return new_node


    def playout(node, fast_policy):
        """ Runs a simulation starting at the node. Returns result/ reward. """
        pass

    def backpropagate(node, reward):
        """ Backpropagates reward from the node to the root. """
