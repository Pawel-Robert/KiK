
import numpy as np

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

class Random_Agent:
    """ Agent making totaly random moves. """
    def __init__(self):
        pass
    def act(self, legal_actions):
        return np.random.choice(legal_actions)

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



class Small_Agent_Explorator:
    """Base class for Agent for a 3x3 board with randomness."""
    def __init__(self, network, epsilon):
        self.network = network
        self.epsilon = epsilon


    def act(self, state_1, state_2, legal_actions, iteration, player):
        """ Choose best action. Returns action and corresponding Q-value """
        st_1 = state_1.flatten()
        st_input_1 = np.array([st_1])
        st_2 = state_2.flatten()
        st_input_2 = np.array([st_2])

        """ With probability epsilon choose random action. """
        if np.random.random_sample() < self.epsilon:
            action = np.random.choice(legal_actions)
            ac = np.zeros(9)
            ac[action] = 1
            ac_input = np.array([ac])
            current_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])
            return action, current_q_value
        else:
            target_action = legal_actions[0]
            ac = np.zeros(9)
            ac[target_action] = 1
            ac_input = np.array([ac])
            return_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])

            """ Search for the action with the highest Q value """
            for action in legal_actions:
                ac = np.zeros(9)
                ac[action] = 1
                ac_input = np.array([ac])

                """Compute Q value using the network."""
                current_q_value = float(self.network.model([st_input_1, st_input_2, ac_input])[0][0])

                """ Compare if the new Q-value is higher to the previously chosen one """
                if player == 1:
                    if current_q_value >= return_q_value:
                        #print(f'current_q_value = {current_q_value}')
                        return_q_value = current_q_value
                        target_action = action

                """ Compare if the new Q-value is lower to the previously chosen one """
                if player == -1:
                    if current_q_value <= return_q_value:
                        return_q_value = current_q_value
                        target_action = action

            return target_action, return_q_value
