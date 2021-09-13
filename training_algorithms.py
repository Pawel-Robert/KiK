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
   #        return nxt_q
            return gamma * (alpha * current_q + (1 - alpha) * nxt_q)
