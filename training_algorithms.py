""" Class with various training algorithms """


class BellmanAlgorithm():
    """ Bellman algorithm. """
    def __init__(self, alpha=0.8, gamma=0.95):
        self.alpha = alpha
        self.gamma = gamma

    def q_value(self, current_q, nxt_q, reward, done):
        """ Computes target q value using Bellman equation. """
        if done:
            return reward
        else:
            return self.gamma * (self.alpha * current_q + (1 - self.alpha) * nxt_q)
