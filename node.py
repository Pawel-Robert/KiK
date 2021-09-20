""" Node class. """

class Node:
    """ Class node. """
    def __init__(self, parent, state):
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def get_value(self):
        """ Returns predicted value of the node. """
        return self.wins/self.visits

    def add_child(self, node):
        self.children.append(node)

