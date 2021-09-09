""" Classes implementing various agents. """

import numpy as np
# from MCTS import MonteCarloTreeSearch


class Small_Agent:
    """Base class for Agent for a 3x3 board (requires flattening of the input). No randomness."""
    def __init__(self, network):
        self.network = network

    def act(self, state, legal_actions, player):
        """ Choose best action. Returns action and corresponding Q-value """
        del player
        q_values = [self.network.evaluate(state, action) for action in legal_actions]
        max_id = np.argmax(q_values)
        return legal_actions[max_id], q_values[max_id]


class Small_Agent_Explorator(Small_Agent):
    """Base class for Agent for a 3x3 board with randomness."""
    def __init__(self, network, epsilon):
        super().__init__(network)
        self.epsilon = epsilon

    def act(self, state, legal_actions, player, iteration=1):
        if np.random.random() < self.epsilon / np.sqrt(iteration):
            action = np.random.choice(legal_actions)
            current_q_value = self.network.evaluate(state, action)
            return action, current_q_value
        return super().act(state, legal_actions, player)


# class Small_MCTS_Agent(Small_Agent):
#     """ Uses Monte Carlo Tree Search algorithm. """
#     def __init__(self, network):
#         super().__init__(network)
#         self.env = env
#         self.mcts = MonteCarloTreeSearch(network, env)
#
#     def act(self, state, legal_actions, player):
#         """ Choose best action. Returns action and corresponding Q-value """
#         action = self.mcts.predict_action(state, player)
#         return action

# class Random_Agent:
#     """ Agent making totaly random moves. """
#     def __init__(self):
#         pass
#     def act(self, observation, legal_actions, player):
#         """ Random move. """
#         del observation
#         del player
#         return np.random.choice(legal_actions), 0

class Heuristic_Agent:
    """ Make a winnig move if possible, block opponent win move otherwise or do a random move. """
    def __init__(self):
        self.height = 3
        self.width = 3
        self.win_cnd = 3

    def check_win(self, board, player):
        """ Function checking if the game is won by some player. Returns boolean. """
        for col in range(self.height):
            for row in range(self.width - self.win_cnd + 1):
                if sum([board[col][row+i] for i in range(self.win_cnd)]) == self.win_cnd*player:
                    return True
        for row in range(self.width):
            for col in range(self.height - self.win_cnd +1):
                if sum([board[col+i][row] for i in range(self.win_cnd)]) == self.win_cnd*player:
                    return True
        for col  in range(self.height - self.win_cnd + 1):
            for row in range(self.width - self.win_cnd +1):
                if sum([board[col+i][row+i] for i in range(self.win_cnd)]) == self.win_cnd*player:
                    return True
        for col in range(self.height - self.win_cnd + 1):
            for row in range(self.width - self.win_cnd + 1):
                diag = sum([board[col+i][row+self.win_cnd-i-1] for i in range(self.win_cnd)])
                if diag == self.win_cnd*player:
                    return True
        return False

    def act(self, state, legal_actions, player):
        """ Check if there is a win action. Then check if the opponent
         has a win action to block. Otherwise make a random move. """
        if len(legal_actions) == 9:
            return 4, 0
        for action in legal_actions:
            temporal_board = np.copy(state)
            y_position = action // self.width
            x_position = action - y_position * self.width
            temporal_board[y_position, x_position] = player
            if self.check_win(temporal_board, player):
                return action, 0
        for action in legal_actions:
            temporal_board = np.copy(state)
            y_position = action // self.width  # bierzemy podłogę z dzielenia
            x_position = action - y_position * self.width  # reszta z dzielenia
            # aktualizujemy stan planszy
            temporal_board[y_position, x_position] = -player
            if self.check_win(temporal_board, - player):
                return action, 0
        return np.random.choice(legal_actions), 0
