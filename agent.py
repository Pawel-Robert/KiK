
import numpy as np

class Small_Agent:
    """Base class for Agent for a 3x3 board (requires flattening of the input). No randomness."""
    def __init__(self, network):
        self.network = network


    def act(self, state, legal_actions, player):
        """ Choose best action. Returns action and corresponding Q-value """
        st = state.flatten()
        st_input = np.array([st])
        return_q_value = - player
        target_action = 0


        """ Loop searching for the action with the highest/lowest Q value."""
        for action in legal_actions:
            ac = np.zeros(9)
            ac[action] = 1
            ac_input = np.array([ac])
            current_q_value = self.network.model([st_input, ac_input])

            """For the first player we are maximasiong the Q value."""
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

class Small_Agent_Explorator:
    """Base class for Agent for a 3x3 board with randomness."""
    def __init__(self, network, epsilon):
        self.network = network
        self.epsilon = epsilon

    def act(self, state, legal_actions, N, player):
        """ Choose best action. Returns action and corresponding Q-value """
        st = state.flatten()
        st_input = np.array([st])

        """ With probability epsilon choose random action. """
        if np.random.random_sample() < self.epsilon:
            action = np.random.choice(legal_actions)
            ac = np.zeros(9)
            ac[action] = 1
            ac_input = np.array([ac])
            current_q_value = float(self.network.model([st_input, ac_input])[0][0])
            return action, current_q_value
        else:
            return_q_value = - player
            target_action = 0

            """ Search for the action with the highest Q value """
            for action in legal_actions:
                ac = np.zeros(9)
                ac[action] = 1
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
