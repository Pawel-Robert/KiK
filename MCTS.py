from numpy import random

class Node:
    """ Class representing nodes in a tree. """
    def __init__(self, parent, state, is_leaf = True):
        self.parent = parent
        self.is_leaf = is_leaf
        self.is_root = False
        self.wins = 0
        self.playouts = 0
        self.state = state
        self.childrens = []


class MonteCarloTreeSearch:
    """ Class representing trees of game states and actions for seraching """
    def __init__(self, network, env, root, state):
        self.network = network
        self.env = env
        self.root = root
        self.nodes = [root]
        self.state = state

    def expand(self, node):
        """ Expands a node, creating its child.  """
        action = random.choice(self.env.legal_actions)
        state = self.env.step(action)
        new_node = Node(node, True, state)
        self.nodes.append(Node(node, True))
        return Node(node, True)

    def simulate(self, node):
        """ Simulates a rollout starting at the node. """
        state = node.state


        while True:
            action = network.predict(state)
            env.step(action)
            if return != 0:
                node.wins_count += 1
                node.visit_count += 1
                break

    def backpropagate(self, node, reward):
        """ Backpropagates results of a rollout made form the node. """
        temp_node = node
        while True:
            if temp_node.is_root:
                break
            temp_node = temp.node.parent
            temp_node.wins_count += reward
            temp_node.visit_count += 1


    def select(self, node):
        """ Selects children starting at the node until a leaf is reached. """
        temp_node = node
        while True:
            temp_node = random.choice(temp_node.childrens)
            if temp_node.is_leaf:
                return temp_node