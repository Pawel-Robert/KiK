""" Class with various training algorithms """

class Algorithm:
    """ Class algorithm. """
    def __init__(self):
        pass

    def q_value(self, current, nxt, reward, alpha, gamma, done):
        """ Do nothing. """
        return current

class BellmanAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()

    def q_value(self, current_q, nxt_q, reward, done, alpha, gamma):
        """ Computes target q value using Bellman equation. """
        # return 1
        if done:
            return reward
        else:
            return gamma * (alpha * current_q + (1 - alpha) * nxt_q)

class BellmanValueAlgorithm(Algorithm):
    """ Variation of the Bellman algorithm for the value network. """
    def __init__(self):
        super().__init__()

    def distribution(self, current_dist, nxt_dist, reward, done, alpha, gamma):
        """ Computes target distribution using Bellman equation. """
        if done:
            distribution = np.zeros(current_dist.shape)
            distribution[action] = 1
            return reward
        else:
            return gamma * (alpha * current_dist + (1 - alpha) * nxt_dist)
