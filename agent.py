# klasa implementująca algorytmy wybierania akcji na podstawie obserwacji oraz modelu

import numpy as np

class Agent:
    """Base class for Agent"""
    def __init__(self, model, legal_actions):
        self.model = model
        self.legal_actions = legal_actions

    def act(self, state):
        """Choose best action. Returns action"""
        for action in self.legal_action:
            x = -1
            y = self.model.predict_value(state, action)
            if x >= y:
                x = y
                target_action = action
        return target_action

# class EpsilonGreedyAgent(Agent):
#     def act(self, state, espilon):
#         if np.random.random() < epsilon:
#             return np.random.choice(self.num_actions)
#         else:
#             return np.argmax(model.predict(state))

class MCTSAgent(Agent):
    """Use Monte Carlo rollouts to explore the game tree"""
    def act(self, state):
        # wykonujemy depth korków w głąb drzewa
        for i in range(depth):
            action = model(observation)
            env.step(action)
        raise NotImplementedError


#funckja wybierająca akcję za pomocą modelu
def choose_action(model, observation):
    # add batch dimension to the observation if only a single example was provided
    # observation = np.expand_dims(observation, axis=0) if single else observation
    logits = model.predict(observation)
    #losujemy akcję, która jest dozwolonym ruchem
    while True:
        action = tf.random.categorical(logits, num_samples=1)
        if env.is_allowed_move(action):
            break
    action = action.numpy().flatten()
    return action

# to jest funkcja rekurencyjna (pytanie: jak ją zatrzymać?)
def search(s, game, nnet):
    if game.gameEnded(s): return -game.gameReward(s)

    if s not in visited:
        visited.add(s)
        P[s], v = nnet.predict(s)
        return -v

    max_u, best_a = -float("inf"), -1
    for a in game.getValidActions(s):
        u = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
        if u > max_u:
            max_u = u
            best_a = a
    a = best_a

    sp = game.nextState(s, a)
    v = search(sp, game, nnet)

    Q[s][a] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
    N[s][a] += 1
    return -v
