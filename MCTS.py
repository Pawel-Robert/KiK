import numpy as np
from math import pow

class Small_Agent:
    """Base class for Agent for a 3x3 board (requires flattening of the input). No randomness."""
    def __init__(self, network):
        self.network = network


    def act(self, state_1, state_2, legal_actions, player):
        """ Choose best action. Returns action and corresponding Q-value """
        st_1 = state_1.flatten()
        st_input_1 = np.array([st_1])
        st_2 = state_2.flatten()
        st_input_2 = np.array([st_2])
        ac = np.zeros(9)
        target_action = legal_actions[0]
        ac[target_action] = 1
        ac_input = np.array([ac])
        return_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])
        # print(return_q_value)

        """ Loop searching for the action with the highest/lowest Q value."""
        for action in legal_actions:
            """ Using the network compute Q value of the action. """
            ac = np.zeros(9)
            ac[action] = 1
            ac_input = np.array([ac])
            current_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])

            """For the first player we are maximizing the Q value."""
            if player == 1:
                if current_q_value >= return_q_value:
                    return_q_value = current_q_value
                    target_action = action

            """For the second player we are minimising the Q value."""
            if player == -1:
                if current_q_value <= return_q_value:
                    return_q_value = current_q_value
                    target_action = action

        return target_action, return_q_value

class Node:
    """ Class representing nodes in a tree. Each node corresponds to a state in the game. """

    def __init__(self, parent, action, state, player, is_leaf=True):
        self.parent = parent
        """ Action taken to create that node. """
        self.action = action
        self.is_leaf = is_leaf
        self.wins_count = 0
        self.children = []
        self.value_sum = 0
        self.visit_count = 1
        self.player = player

        """ State should be a numpy array. """
        """ It is a state of the game which the node represents. """
        self.state = state
        self.width = state.shape[0]
        self.height = state.shape[1]

        # """ List of actions. """
        # self.actions = [i * self.width + j for i in range(self.width) and j in range(self.height)]

    def actions(self):
        list = []
        for i in range(self.height):
            for j in range(self.width):
                if self.state[i, j] == 0:
                    list.append(i * self.width + j)
        return list


class Distribution:
    """ It is a function which asignes to a action its probability."""

    def __init__(self, set):
        self.set = set
        """ Set of probability values for the elements of the set. """
        self.values = []
        for _ in set:
            self.values.append(0.1)

    def set_value(self, action, value):
        """ Set value of an input action. """
        for n in range(len(self.set)):
            if self.set[n] == action:
                self.values[n] = value

    def normalise(self):
        """ Divides all the values by the norm (total mas). """
        for n in range(len(self.values)):
            self.values[n] = self.values[n] / self.norm()
        pass

    def norm(self):
        return sum(a for a in self.values)


class MonteCarloTreeSearch:
    """ Class representing trees of game states and actions for searching """

    def __init__(self, network, env, number_of_simulations=4, temperature=1):
        """ Network predicting the actions. """
        self.network = network
        """ Environment in which we are acting. """
        self.env = env
        """ Collection of nodes created during each bunch of simulations. """
        self.tree = []

        self.agent = Small_Agent(network)
        """ Number of simulations run for each call of the functions predict."""
        self.number_of_simulations = number_of_simulations
        """ Temperature parametrising output probability distribution. """
        self.temperature = temperature

    def predict_q_value(self, state, player, input_action):
        """ We run number of simulations to upgreade the policy. """
        """ It returns better evalueted Q value for the Bellmans equation. """
        """ We need to take into account the initial value of the Q function, which we want to make better. """
        

        """ Clear the tree. """
        self.tree = []

        """ Create a root in the starting position 'state'. """
        root = Node(None, None, state, player, True)
        self.tree.append(root)

        """ Run the simulations in 4 steps: select, expand, evaluate and backpropagate. """
        for i in range(self.number_of_simulations):
            # print(f'Simulation number = {i + 1}')
            curr_node = self.select(root)
            self.expand(curr_node)
            """ Result of the fast simulation. """
            reward = self.simulate(curr_node)
            # print(f'reward = {reward}')
            self.backpropagate(curr_node, reward)

            # print(len(self.tree))
        # # self.draw_tree(root)
        # for node in self.tree:
        #     print(node.children)
        #     self.render(node)

        # """ Distribution on the set of actions depends on the visit count. """
        # distribution = Distribution(self.env.legal_actions())
        # for node in root.children:
        #     distribution.set_value(node.action, pow(node.visit_count, self.temperature))
        #     distribution.normalise()

        """" We need to return the new value of the Q function for the input action. """
        """ First we need to look which child gives that action. """
        for node in root.children:
            if node.action == input_action:
                q_value = node.value_sum/node.visit_count

        return q_value


    def predict_action(self, state, player):
        """ We run number of simulations to upgreade the policy. """
        """ It searches for the best action. """


        """ Clear the tree. """
        self.tree = []

        """ Create a root in the starting position 'state'. """
        root = Node(None, None, state, player, True)
        self.tree.append(root)

        """ Run the simulations in 4 steps: select, expand, evaluate and backpropagate. """
        for i in range(self.number_of_simulations):
            # print(f'Simulation number = {i + 1}')
            curr_node = self.select(root)
            self.expand(curr_node)
            """ Result of the fast simulation. """
            reward = self.simulate(curr_node)
            # print(f'reward = {reward}')
            self.backpropagate(curr_node, reward)

            # print(len(self.tree))
        # # self.draw_tree(root)
        # for node in self.tree:
        #     print(node.children)
        #     self.render(node)

        # """ Distribution on the set of actions depends on the visit count. """
        # distribution = Distribution(self.env.legal_actions())
        # for node in root.children:
        #     distribution.set_value(node.action, pow(node.visit_count, self.temperature))
        #     distribution.normalise()

        if root.children:
            target_node = self.arg_max_child(root)
            action = target_node.action
        else:
            action = None
        # return distribution
        return action

    def select(self, node):
        """ Selects children starting at the node until a leaf is reached. """
        """ Selection maximalise Q value function. """
        temp_node = node
        while True:
            # print(temp_node.state)
            if not temp_node.children:
                return temp_node
            """ We choose a random child of temp node aa=ccording to the q_value. """
            temp_node = self.arg_max_child(temp_node)

    def arg_max_child(self, node):
        """ Loop searching for the action with the highest Q value among nodes form the list of children. """
        target_node = node.children[0]
        target_q_value = target_node.value_sum / target_node.visit_count

        for node in node.children[1:]:
            q_value = node.value_sum / node.visit_count
            if q_value > target_q_value:
                target_q_value = q_value
                target_node = node
        return target_node

    def expand(self, node):
        """ Expands a node, creating its child.  """

        """ Set the environment according to the node. """
        self.env.board = np.copy(node.state)
        self.env.player = node.player

        """ If there is a possible action mark that node is no longer a leaf. """
        # print(node.state)
        if self.env.legal_actions():
            node.is_leaf = False

        """ For any legal action create a child of the node. """
        for action in self.env.legal_actions():
            """ Make a step in the environment. """
            state, reward, done, info = self.env.step(action)
            """ Create a new node, which is a child of the node. """
            new_node = Node(node, action, state, -node.player, True)
            """ Mark the new_node as a child of the node. """
            node.children.append(new_node)
            """ Put the new node in the tree. """
            self.tree.append(new_node)
            """ Set the environment according to the node. """
            self.env.board = np.copy(node.state)
            self.env.player = node.player

        return node.children

    def simulate(self, node):
        """ Simulates a rollout starting at the node. """
        self.env.board = np.copy(node.state)
        self.env.player = node.player
        state_1 = self.env.player_board(1)
        state_2 = self.env.player_board(-1)

        """ Play until the game is over. Return reward. """
        while True:
            """ Using the network choose an action"""
            # print(node.actions())
            # print(node.state)
            if self.env.legal_actions():
                action, q_value = self.agent.act(state_1, state_2, self.env.legal_actions(), 1)
                """ Make a step in the environment. """
                next_observation, reward, done, info = self.env.step(action)
                """ If done collect results and assign them to the node. """
            else:
                return 0
            if done:
                node.wins_count += reward
                node.visit_count += 1
                return reward

    def backpropagate(self, node, reward):
        """ Backpropagates results of a rollout made form the node. """
        temp_node = node
        while True:
            if temp_node.parent == None:
                break
            temp_node = temp_node.parent
            temp_node.wins_count += reward
            temp_node.visit_count += 1
            """ For each node in the selected path add value of the ultimate state. """
            state_1 = self.env.player_board(1).flatten()
            state_2 = self.env.player_board(-1).flatten()
            st_input_1 = np.array([state_1])
            st_input_2 = np.array([state_2])
            ac = np.zeros(9)
            ac[node.action] = 1.
            ac_input = np.array([ac])
            temp_node.value_sum += self.network.model([st_input_1, st_input_2, ac_input])

    def draw_tree(self, root):
        queue = [root]
        while True:
            node = queue.pop(0)
            print(queue)
            print(node.actions())
            print(node.children)
            print(f'node of state:')
            for child in node.children:
                queue.append(child)
                self.render(child)

    def render(self, node):
        self.env.board = node.state
        self.env.render()
