from kik_env import KiKEnv
import numpy as np

width = 6
height = 6

env = KiKEnv(width, height)

# trajectory = env.random_play()

# print(trajectory)

env.human_vs_human_play()
buffer_size = 100
time_limit = 100

runner = Runner(Agent, model, env, buffer_size, time_limit)

runner.run_one_episode()

def add_trajectory(self, trajectory, alpha, gamma):
    """Adds trajectory to buffer"""
    # self.trajectories.append(trajectory)
    # lista pomocnicza
    b = []
    # a = (observation, action, reward, done)
    for a in trajectory:
        if b != []:
            # stan poprzedni
            state = b[0]
            modified_action = b[1]
            self.data[0].append([state, modified_action])

            target_Q_value = self.alfa * b[0] + (1 - self.alpha) * a[0] + b[2] * self.gamma
            self.data[1].append(target_Q_value)
        # zachowujemy do kolejnego kroku jako wartość poprzednią
        b = a
    pass

def random_play(self, moves_limit=None):
    self.reset()
    traj = []
    t = 0
    while True:
        # losujemy ruch i sprawdzamy, czy jest dozwolony
        while True:
            x = np.random.random_integers(0, self.width - 1)
            y = np.random.random_integers(0, self.height - 1)
            action = x + y * self.width
            if self.is_allowed_move(action):
                break
        state, reward, done, info = self.step(action)
        # print(state,action)
        traj.append([copy(state), action, reward, done])
        # print(trajectory)
        if done:
            break
        if moves_limit is not None:
            t += 1
            if t == moves_limit:
                break
        if not self.legal_actions():
            break
    return traj