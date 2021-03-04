import numpy as np

class KiKEnv():
    #metadata = {'render.modes': ['human']}

    #inicjując podajemy rozmiary planszy.
    def __init__(self, height, width, player = 1):

        # plasza o rozmiarach heightx width na której stawiamy na przemian znaki
        self.height = height
        self.width = width
        self.board = np.zeros((height, width))
        self.reward = 0
        self.player = player
        pass

    #spawdzamy, czy na plaszy pojawił się kształt dający zwycięstwo
    def check_win(self):
        for a in range(self.height-5):
            for b in range(self.width-5):
                if self.compute_filter(a,b):
                    return True
        return False



    # sprawdzamy zwycięstwo lokalnie w kwadracie 5x5 o górnym lewym wierzchołku w pozycji (a,b)
    def compute_filter(self,a,b):
        horizontal = 0
        vertical = 0
        slantwise = 0
        counter_slantwise = 0
        for n in range(5):
            horizontal = horizontal + self.board[a+n,b]
            vertical = vertical + self.board[a,b+n]
            slantwise = slantwise + self.board[a+n,b+n]
            counter_slantwise = counter_slantwise + self.board[a+4-n,b+n]
        if max(horizontal, vertical, slantwise, counter_slantwise) == 5:
            self.reward = 1
            return True
        elif min(horizontal, vertical, slantwise, counter_slantwise) == -5:
            self.reward = -1
            return True
        return False

    # funkcja spawdzająca, czy wykonanie danej akcji jest poprawne, to znaczy czy pole nie jest już zajęte
    def is_allowed_move(self,a,b):
        if self.board[a,b] == 0:
            return True
        else:
            return False

    # zwracamy listę dostępnych ruchów w obecnym stanie rozgrywki
    def legal_actions(self):
        list = []
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i,j] == 0:
                    list.append([i,j])
        return list


    # funckja step
    def step(self, action):
        player = action[2]
        self.board[action[0],action[1]] = player
        observation = self.board
        reward = self.reward
        done = False
        if self.check_win():
            done = True
            reward = self.reward
        return observation, reward, done, {}

    # przywrócenie ustawień początkowych gry
    def reset(self):
        self.board = np.zeros((self.height,self.width))
        self.reward = 0
        pass

    # wypisanie aktualnego stanu gry
    def render(self):
        print(self.board)
        pass
