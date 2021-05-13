
import numpy as np



class Random_Agent:
    """ Agent making totaly random moves. """
    def __init__(self):
        pass
    def act(self, legal_actions):
        return np.random.choice(legal_actions)

class Heuristic_Agent:
    ''' Heuristic play: make a winnig move if possible, block opponent winning move otherwise or do a random move'''
    def __init__(self, env):
        self.height = env.height
        self.width = env.width
        self.winning_condition = env.winning_condition

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



class Big_Agent_Explorator:
    """Base class for Agent for a 3x3 board with randomness."""
    def __init__(self, network, epsilon, height, width):
        self.network = network
        self.epsilon = epsilon
        self.height = height
        self.width = width

    def act(self, state, legal_actions, iteration, player):
        """ Choose best action. Returns action and corresponding Q-value """

        st_input = np.array([state])

        """ With probability epsilon choose random action. """
        if np.random.random_sample() < (self.epsilon/(iteration+1)):
            action = np.random.choice(legal_actions)
            ac = np.zeros((self.height, self.width))
            a = action % self.width
            b = action // self.width
            ac[a,b] = 1
            ac_input = np.array([ac])
            current_q_value = float(self.network.model([st_input, ac_input])[0][0])
            return action, current_q_value
        else:
            target_action = legal_actions[0]
            ac = np.zeros((self.height, self.width))
            a = target_action % self.width
            b = target_action // self.width
            ac[a, b] = 1
            ac_input = np.array([ac])
            return_q_value = float(self.network.model([st_input, ac_input])[0][0])

            """ Search for the action with the highest Q value """
            for action in legal_actions:
                ac = np.zeros((self.height, self.width))
                a = action % self.width
                b = action // self.width
                ac[a, b] = 1
                ac_input = np.array([ac])

                """Compute Q value using the network."""
                current_q_value = float(self.network.model([st_input, ac_input])[0][0])

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


class Big_Agent:
    """Base class for Agent for a 3x3 board with randomness."""
    def __init__(self, network, height, width):
        self.network = network
        self.height = height
        self.width = width

    def act(self, state, legal_actions, player):
        """ Choose best action. Returns action and corresponding Q-value """
        st_input = np.array([state])
        target_action = legal_actions[0]
        ac = np.zeros((self.height, self.width))
        a = target_action % self.width
        b = target_action // self.width
        ac[a, b] = 1
        ac_input = np.array([ac])
        return_q_value = float(self.network.model([st_input, ac_input])[0][0])

        """ Search for the action with the highest Q value """
        for action in legal_actions:
            ac = np.zeros((self.height, self.width))
            a = action % self.width
            b = action // self.width
            ac[a, b] = 1
            ac_input = np.array([ac])

            """Compute Q value using the network."""
            current_q_value = float(self.network.model([st_input, ac_input])[0][0])

            """ Compare if the new Q-value is higher to the previously chosen one """
            if player == 1:
                if current_q_value >= return_q_value:
                    # print(f'current_q_value = {current_q_value}')
                    return_q_value = current_q_value
                    target_action = action

            """ Compare if the new Q-value is lower to the previously chosen one """
            if player == -1:
                if current_q_value <= return_q_value:
                    return_q_value = current_q_value
                    target_action = action

        return target_action, return_q_value


# class Big_Agent():
#     """Base class for Agent on any board"""
#     def __init__(self, network, height, width):
#         self.network = network
#         self.height = height
#         self.width = width
#     """Base class for Agent"""
#     def act(self, state, legal_actions):
#         """Choose best action. Returns action"""
#         st_imput = np.array([st])
#         max_q_value = -1
#         target_action = 0
#         for action in legal_actions:
#             ac_input = np.zeros(self.height, self.width)
#             ac[action // self.width, action % self.width] = 1
#             current_q_value = self.network.model([state, ac_input])
#             if current_q_value >= max_q_value:
#                 max_q_value = current_q_value
#                 target_action = action
#         return target_action, max_q_value


# class EpsilonGreedyAgent(Agent):
"""Add some randomness to the exploration"""
#     def act(self, state, espilon):
#         if np.random.random() < epsilon:
#             return np.random.choice(self.num_actions)
#         else:
#             return np.argmax(model.predict(state))

# class MCTSAgent():
#     """Use Monte Carlo rollouts to explore the game tree"""
#     def act(self, state):
#         # wykonujemy depth korków w głąb drzewa
#         for i in range(depth):
#             action = model(observation)
#             env.step(action)
#         raise NotImplementedError



# # to jest funkcja rekurencyjna (pytanie: jak ją zatrzymać?)
# def search(s, game, nnet):
#     if game.gameEnded(s): return -game.gameReward(s)
#
#     if s not in visited:
#         visited.add(s)
#         P[s], v = nnet.predict(s)
#         return -v
#
#     max_u, best_a = -float("inf"), -1
#     for a in game.getValidActions(s):
#         u = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
#         if u > max_u:
#             max_u = u
#             best_a = a
#     a = best_a

    # sp = game.nextState(s, a)
    # v = search(sp, game, nnet)
    #
    # Q[s][a] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
    # N[s][a] += 1
    # return -v
