class Agent:
    """Base class for Agent"""
    def __init__(self, model, board_size):
        self.model = model
        self.board_size = board_size

    def act(self, state):
        """Choose best action. Returns action"""
        raise NotImplementedError


class MCTSAgent(Agent):
    def act(self, state):
        raise NotImplementedError