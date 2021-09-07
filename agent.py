
import numpy as np
from MCTS import MonteCarloTreeSearch


class Small_Agent:
    """Base class for Agent for a 3x3 board (requires flattening of the input). No randomness."""
    def __init__(self, network):
        self.network = network

    def act(self, state, legal_actions, player):
        """ Choose best action. Returns action and corresponding Q-value """
        action = np.zeros(9)
        """ Start with first possible action before running the loop. """
        target_action = legal_actions[0]
        action[target_action] = 1
        return_q_value = self.network.evaluate(state, action)
        # print(return_q_value)

        """ Loop searching for the action with the highest/lowest Q value."""
        for action in legal_actions:
            """ Using the network compute Q value of the action. """
            action_input = np.zeros(9)
            action_input[action] = 1
            current_q_value = self.network.evaluate(state, action_input)

            """For the first player we are maximizing the Q value."""
            if player == 1:
                if current_q_value >= return_q_value:
                    return_q_value = current_q_value
                    target_action = action

            """For the second player we are minimising the Q value."""
            # This is redundant, as the network acts always as the first player.
            if player == -1:
                if current_q_value <= return_q_value:
                    return_q_value = current_q_value
                    target_action = action

        return target_action, return_q_value

class Small_Agent_Explorator(Small_Agent):
    """Base class for Agent for a 3x3 board with randomness."""
    def __init__(self, network, epsilon):
        super().__init__(network)
        """ Probability of choosing random action. """
        self.epsilon = epsilon


    def act(self, state, legal_actions, player, iteration=1):
        if np.random.random() < self.epsilon / np.sqrt(iteration):
            action = np.random.choice(legal_actions)
            action_input = np.zeros(9)
            action_input[action] = 1
            current_q_value = self.network.evaluate(state, action_input)
            return action, current_q_value
        else:
            return super().act(state, legal_actions, player)

    # def act_old(self, state_1, state_2, legal_actions, player, iteration=1):
    #     """ Choose best action. Returns action and corresponding Q-value """
    #     st_input_1 = self.prepare_state(state_1)
    #     st_input_2 = self.prepare_state(state_2)
    #
    #     """ With probability epsilon/sqrt(iteration) choose random action. """
    #     if np.random.random() < self.epsilon/np.sqrt(iteration):
    #         action = np.random.choice(legal_actions)
    #         ac = np.zeros(9)
    #         ac[action] = 1
    #         ac_input = np.array([ac])
    #         current_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])
    #         return action, current_q_value
    #     else:
    #         target_action = legal_actions[0]
    #         ac = np.zeros(9)
    #         ac[target_action] = 1
    #         ac_input = np.array([ac])
    #         return_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])
    #
    #         """ Search for the action with the highest Q value """
    #         for action in legal_actions:
    #             ac = np.zeros(9)
    #             ac[action] = 1
    #             ac_input = np.array([ac])
    #
    #             """Compute Q value using the network."""
    #             current_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])
    #
    #             """ Compare if the new Q-value is higher to the previously chosen one """
    #             if player == 1:
    #                 if current_q_value >= return_q_value:
    #                     #print(f'current_q_value = {current_q_value}')
    #                     return_q_value = current_q_value
    #                     target_action = action
    #
    #             """ Compare if the new Q-value is lower to the previously chosen one """
    #             if player == -1:
    #                 if current_q_value <= return_q_value:
    #                     return_q_value = current_q_value
    #                     target_action = action
    #
    #         return target_action, return_q_value


class Small_MCTS_Agent:
    """Base class for Agent for a 3x3 board (requires flattening of the input). """
    """ Uses Monte Carlo Tree Search algorithm. """
    """" Sends back only action! """
    def __init__(self, network, env):
        self.network = network
        self.env = env
        """ MCTS algorithm. """
        self.mcts = MonteCarloTreeSearch(network, env)


    def act(self, state, player):
        """ Choose best action. Returns action and corresponding Q-value """
        action = self.mcts.predict_action(state, player)
        return action



class Random_Agent:
    """ Agent making totaly random moves. """
    def __init__(self):
        pass
    def act(self, observation, legal_actions, player):
        del observation
        del player
        return np.random.choice(legal_actions), 0

class Heuristic_Agent:
    ''' Heuristic play: make a winnig move if possible, block opponent winning move otherwise or do a random move'''
    def __init__(self):
        self.height = 3
        self.width = 3
        self.winning_condition = 3

    def check_win(self, board, player):
        """ Function checking if the game is won by some player. Returns boolean. """
        for col in range(self.height):
            for row in range(self.width - self.winning_condition + 1):
                if sum([board[col][row+i] for i in range(self.winning_condition)]) == self.winning_condition*player:
                    return True
        for row in range(self.width):
            for col in range(self.height - self.winning_condition +1):
                if sum([board[col+i][row] for i in range(self.winning_condition)]) == self.winning_condition*player:
                    return True
        for col  in range(self.height - self.winning_condition + 1):
            for row in range(self.width - self.winning_condition +1):
                if sum([board[col+i][row+i] for i in range(self.winning_condition)]) == self.winning_condition*player:
                    return True
        for col in range(self.height - self.winning_condition + 1):
            for row in range(self.width - self.winning_condition + 1):
                if sum([board[col+i][row +self.winning_condition-i-1] for i in range(self.winning_condition)]) == self.winning_condition*player:
                    return True
        return False

    def act(self, state, legal_actions, player):
        """ Check if there is a winning action. """
        if len(legal_actions) == 9:
            return 4, 0
        for action in legal_actions:
            temporal_board = np.copy(state)
            y_position = action // self.width
            x_position = action - y_position * self.width
            temporal_board[y_position, x_position] = player
            if self.check_win(temporal_board, player):
                return action, 0
        """ Check if the oponent has a winning action to block. """
        for action in legal_actions:
            temporal_board = np.copy(state)
            y_position = action // self.width  # bierzemy podłogę z dzielenia
            x_position = action - y_position * self.width  # reszta z dzielenia
            # aktualizujemy stan planszy
            temporal_board[y_position, x_position] = -player
            if self.check_win(temporal_board, - player):
                return action, 0
        """ Make a random move. """
        return np.random.choice(legal_actions), 0
