class Runner:
    def __init__(self, board_size, agent_class, model, env):
        """The main class to run the training loop"""
        self.board_size = board_size
        self.model = model
        self.agent = agent_class(self.model, self.board_size)