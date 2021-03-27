from copy import copy

import numpy as np


class KiKEnv():
    #metadata = {'render.modes': ['human']}

    # inicjując podajemy rozmiary planszy.
    # width - szerokość
    # height - wysokość
    # pusta plansza width = 3 na heigth = 2 ma postać:
    # 0 0 0
    # 0 0 0
    def __init__(self, width, height, winning_condition, player = 1):

        # plasza o rozmiarach heightx width na której stawiamy na przemian znaki
        self.height = height
        self.width = width
        self.winning_condition = winning_condition
        self.board = np.zeros((height, width))
        self.reward = 0
        # self.player to gracz aktualnie wykonujący ruch
        # graczem rozpoczynającym grę jest gracz podany w parametrze player
        self.player = player
        self.q_network = None

    #spawdzamy, czy na plaszy pojawił się kształt dający zwycięstwo
    def check_win(self):
        for a in range(self.height-self.winning_condition+1):
            for b in range(self.width-self.winning_condition+1):
                if self.compute_filter(a,b):
                    return True
        return False


    # sprawdzamy zwycięstwo lokalnie w kwadracie 5x5 o górnym lewym wierzchołku w pozycji (a,b)
    def compute_filter(self,a,b):

        set_of_line_sums = set()
        for col in range(a, a+self.winning_condition):
            set_of_line_sums.add(sum([self.board[col][b+i] for i in range(self.winning_condition)]))
        for row in range(b, b + self.winning_condition):
            set_of_line_sums.add(sum([self.board[a+i][row] for i in range(self.winning_condition)]))

        set_of_line_sums.add(sum([self.board[a + i][b+i] for i in range(self.winning_condition)]))
        set_of_line_sums.add(sum([self.board[a + i][b+self.winning_condition-i-1] for i in range(self.winning_condition)]))

        sum_integers = {int(val) for val in set_of_line_sums}

        print(f' set = {sum_integers}')

        if self.winning_condition in sum_integers:
            self.reward = 1
        elif -self.winning_condition in sum_integers:
            self.reward = -1

        return self.reward

        #
        # horizontal = 0
        # vertical = 0
        # slantwise = 0
        # counter_slantwise = 0
        # for n in range(self.winning_condition):
        #     for
        #     horizontal = horizontal + self.board[a+n,b]
        #     vertical = vertical + self.board[a,b+n]
        #     slantwise = slantwise + self.board[a+n,b+n]
        #     counter_slantwise = counter_slantwise + self.board[a+self.winning_condition-1-n,b+n]
        # if max(horizontal, vertical, slantwise, counter_slantwise) == self.winning_condition:
        #     self.reward = 1
        #     return True
        # elif min(horizontal, vertical, slantwise, counter_slantwise) == - self.winning_condition:
        #     self.reward = -1
        #     return True
        # return False

    # funkcja spawdzająca, czy wykonanie danej akcji jest poprawne, to znaczy czy pole nie jest już zajęte
    def is_allowed_move(self, action):
        a = action % self.width
        b = action // self.width
        if self.board[b,a] == 0:
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


    # funckja step: bierze na wejściu numer pola, a na wyjściu zwraca obserwację itp.
    def step(self, action):
        # na podstawie numeru pola podanego w parametrze action obliczamy jego współrzędne
        y_position = action // self.width  # bierzemy podłogę z dzielenia
        x_position = action % self.width  # reszta z dzielenia
        # aktualizujemy stan planszy
        self.board[y_position, x_position] = self.player
        state = self.board
        reward = 0
        done = False
        if self.check_win():
            done = True
            reward = 1
        self.player = - self.player
        return state, reward, done, {}

    # przywrócenie ustawień początkowych gry
    def reset(self):
        self.board = np.zeros((self.height,self.width))
        self.reward = 0
        return self.board

    # wypisanie aktualnego stanu gry
    def render(self):
        for x in range(self.board.shape[0]):
            print('|', end='')
            for y in range(self.board.shape[1]):
                val = self.board[x][y]
                if val == 1.:
                    print(' o ', end='')
                elif val == -1.:
                    print(' x ', end='')
                else:
                    print(' . ', end='')
            print('|')
        pass

    def check_if_in_range(self,a,b):
        if a <= self.width and a >=1 and b <= self.height and b >=1:
            return True
        else:
            return False

    # funckja umożliwiająca grę we dwóch graczy
    def human_vs_human_play(self):
        #pygame.init()
        #pygame.mixer.init()
        #pygame.mixer.music.load('song.mp3')
        #pygame.mixer.music.play()
        print()
        print()
        print('WITAMY W GRZE!')
        print()
        print('   |/ . |/ ' )
        print('   |\ | |\ ')
        print()
        print('Zasady gry: gracze na przemian stawiają swoje znaki na planszy. Wygrywa gracz,')
        print('który jako pierwszy postawi pięć znaków w rzędzie, kolumnie bądź skośnie.')
        print('Znak gracza rozpoczynającego to liczba 1, drugi gracz dysponuje liczbą -1.')
        print(f'Plansza ma rozmiary {self.width} na {self.height}.')
        print()
        self.reset()
        while True:
            print('Aktualny stan planszy:')
            print()
            self.render()
            print()
            print('Podaj współrzędne pola.')

            # pętla pobierania ruchu oraz sprawdzania, czy ruch jest dozwolony
            while True:
                a, b = input().split()
                # a = input()
                # b = input()
                if a.isnumeric() and b.isnumeric() and self.check_if_in_range(int(a),int(b)):
                    if self.is_allowed_move((int(a) - 1) + (int(b) - 1) * self.width):
                        break
                    else:
                        print('Ruch niedozwolony - pole jest zajęte! Podaj inne pole.')
                else:
                    print(f'Nie ma takiego pola. Podaj dwie liczby naturalne w przedziałach 'f'(1,{self.width}) oraz (1,{self.height}) ')

            # wykonujemy ruch
            action = (int(a) - 1) + (int(b) - 1) * self.width
            state, reward, done, info = self.step(action)
            # sprawdzamy, czy ktoś już wygrał
            if done:
                    self.render()
                    print()
                    print(f'Wygrał gracz {-self.player}.')
                    break

    def random_play(self, moves_limit = None):
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
            #print(trajectory)
            if done:
               break
            if moves_limit is not None:
                t += 1
                if t == moves_limit:
                    break
            if not self.legal_actions():
                break
        return traj


