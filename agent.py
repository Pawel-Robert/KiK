""" Classes implementing various agents. """

import numpy as np


class Agent:
    """Base class for Agent. No randomness."""

    def __init__(self, network):
        self.network = network

    def act(self, state, legal_actions, iteration=1):
        """ Choose best action. Returns action and corresponding Q-value """
        del iteration
        q_values = self.network.evaluate_on_batch(state, legal_actions)
        max_id = np.argmax(np.array([q_values]))
        return legal_actions[max_id], q_values[max_id][0]


class AgentExplorator(Agent):
    """Base class for Agent with randomness."""

    def __init__(self, network, epsilon):
        super().__init__(network)
        self.epsilon = epsilon

    def act(self, state, legal_actions, iteration=1):
        """ As in super with additional randomness. """
        if np.random.random() < self.epsilon / np.sqrt(iteration + 1):
            action = np.random.choice(legal_actions)
            current_q_value = self.network.evaluate(state, action)
            return action, current_q_value
        return super().act(state, legal_actions)
