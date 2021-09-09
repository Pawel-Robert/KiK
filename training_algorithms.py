""" Class with various trainig algorithms """


class Algorithms:
    """ Class algorithm. """
    def __init__(self):
        pass

    def q_value(self, current, nxt, reward, alpha,):
        """ Do nothing. """
        return current

class BellmanAlgorithm:
    def __init__(self):
        super().__init__()

    def q_value(self, current, nxt, reward, alpha, gamma, done):
        """ Computes target q value using Belmans equation. """
        if done:
            return gamma * reward
        else:
            return alpha * current + (1 - alpha) * nxt
