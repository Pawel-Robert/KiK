from copy import copy
# from datetime import datetime
import numpy as np
import time
#import tensorflow as tf


class KiKEnv():
    #metadata = {'render.modes': ['human']}

    def __init__(self, width, height, winning_condition, player = 1):

        """ Size of the board. """
        self.height = height
        self.width = width
        """ Number of moves in a line necessary to win. """
        self.winning_condition = winning_condition
        """ Board initialised as empty. """
        self.board = np.zeros((height, width))

        self.reward = 0
        self.player = player

    def check_win(self, board):
        """ Function checking if the game is won by some player. Returns boolean. """
        for col in range(self.height):
            for row in range(self.width - self.winning_condition + 1):
                if sum([board[col][row+i] for i in range(self.winning_condition)]) == self.winning_condition*self.player:
                    return True
        for row in range(self.width):
            for col in range(self.height - self.winning_condition +1):
                if sum([board[col+i][row] for i in range(self.winning_condition)]) == self.winning_condition*self.player:
                    return True
        for col  in range(self.height - self.winning_condition + 1):
            for row in range(self.width - self.winning_condition +1):
                if sum([board[col+i][row+i] for i in range(self.winning_condition)]) == self.winning_condition*self.player:
                    return True
        for col in range(self.height - self.winning_condition + 1):
            for row in range(self.width - self.winning_condition + 1):
                if sum([board[col+i][row +self.winning_condition-i-1] for i in range(self.winning_condition)]) == self.winning_condition*self.player:
                    return True
        return False


    def is_allowed_move(self, action):
        """ Function checking if a given move is legal. """
        a = action % self.width
        b = action // self.width
        if self.board[b,a] == 0:
            return True
        else:
            return False

    def legal_actions(self):
        """ Returns a list of legal actions in a current state of the game. """
        list = []
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i,j] == 0:
                    list.append(i*self.height+j)
        return list

    # def compute_coordinates(self, action):
    #     """ Recompute the coordinates on the board based on the action. """
    #
    #     return y_position, x_position

    def step(self, action):
        """" Main step function. Executes action and returns board, reward, done and info."""
        y_position = action // self.width
        x_position = action - y_position * self.width
        """ Change board at position action. """
        self.board[y_position, x_position] = self.player

        self.reward = 0
        done = False

        """ Check winning condition. """
        if self.check_win(self.board):
            done = True
            self.reward = self.player

        """ Swap the player. """
        self.player = - self.player

        return self.board, self.reward, done, {}

    def reset(self):
        """ Return initial conditions in the environement. """
        self.board = np.zeros((self.height,self.width))
        self.reward = 0

        return self.board

    def render(self):
        """ Print the board. """
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


    def print_game_menu(self):
        """ Starting menu in the game """
        print('\n\nWITAMY W GRZE!\n')
        print('   |/ . |/ ')
        print('   |\ | |\ \n')
        print('Zasady gry: gracze na przemian stawiają swoje znaki na planszy. Wygrywa gracz,')
        print('który jako pierwszy postawi pięć znaków w rzędzie, kolumnie bądź skośnie.')
        print('Znak gracza rozpoczynającego to liczba 1, drugi gracz dysponuje liczbą -1.')
        print(f'Plansza ma rozmiary {self.width} na {self.height}.\n')
        pass

    def human_input(self):
        """ Human action input. """
        while True:
            print('\nPodaj współrzędne pola.')
            while True:
                try:
                    a, b = input().split()
                    break
                except:
                    print(
                        f'Zły format danych. Podaj dwie liczby naturalne w przedziałach 'f'(1,{self.width}) oraz (1,{self.height}).')
                if a.isnumeric() and b.isnumeric() and self.check_if_in_range(int(a), int(b)):
                    if self.is_allowed_move((int(a) - 1) + (int(b) - 1) * self.width):
                        break
                    else:
                        print('Ruch niedozwolony - pole jest zajęte! Podaj inne pole.')
                else:
                    print(
                        f'Nie ma takiego pola. Podaj dwie liczby naturalne w przedziałach 'f'(1,{self.width}) oraz (1,{self.height}) ')
            action = (int(a) - 1) + (int(b) - 1) * self.width
            print(type(action))
            return action


    def main_loop(self, mod, agent, network):
        """ Main game play loop."""
        while True:
            if not self.legal_actions():
                break
            print('Aktualny stan planszy:\n')
            self.render()
            if mod == 'HvH':
                """ Take human action. """
                action = self.human_input()
            elif mod == 'HvAI':
                if self.player == 1:
                    """ Take human action. """
                    action = self.human_input()
                else:
                    """ Take AI action. """
                    action, q_value = agent.act(self.board, self.legal_actions(), self.player)
                    print(f'Komputer wykonał ruch o wartości {q_value}')
            else:
                action, q_value = agent.act(self.board, self.legal_actions(), self.player)
                print(f'Komputer wykonał ruch o wartości {q_value}')

            """ Make a move. """
            state, reward, done, info = self.step(action)

            """ Ckech if someone already won. """
            if done:
                self.render()
                print()
                print(f'Wygrał gracz {-self.player}.')
                time.sleep(2)
                break

    def game_play(self, agent, network):
        """" Game play! """
        self.print_game_menu()
        while True:
            print("Wybierz mod gry: HvH, HvAI, AIvAI")
            mod = input()
            self.reset()
            """ Main game loop. """
            self.main_loop(mod, agent, network)

            print("Czy chesz zagrać jeszcze raz? (T/N)")
            if input() == 'N':
                break


    def random_play(self, moves_limit = None):
        """ Random self-play. """
        self.reset()
        traj = []
        t = 0
        while True:
            while True:
                x = np.random.random_integers(0, self.width - 1)
                y = np.random.random_integers(0, self.height - 1)
                action = x + y * self.width
                if self.is_allowed_move(action):
                    break
            state, reward, done, info = self.step(action)
            traj.append([copy(state), action, reward, done])
            if done:
               break
            if moves_limit is not None:
                t += 1
                if t == moves_limit:
                    break
            if not self.legal_actions():
                break
        return traj

    def heuristic_agent(self):
        ''' Heuristic play: make a winnig move if possible, block opponent winning move otherwise or do a random move'''
        # sprawdzamy czy mamy ruch wygrywający
        for action in self.legal_actions():
            temporal_board = self.board
            y_position = action // self.width  # bierzemy podłogę z dzielenia
            x_position = action - y_position * self.width  # reszta z dzielenia
            # aktualizujemy stan planszy
            temporal_board[y_position, x_position] = self.player
            if self.check_win(temporal_board):
                return self.step(action)
        # sprawdzamy, czy przeciwnik ma ruch wygrywający
        for action in self.legal_actions():
            temporal_board = self.board
            y_position = action // self.width  # bierzemy podłogę z dzielenia
            x_position = action - y_position * self.width  # reszta z dzielenia
            # aktualizujemy stan planszy
            temporal_board[y_position, x_position] = -self.player
            if self.check_win():
                return self.step(action)
        # jak nie, to robimt ruch losowy
        action = np.random.choice(self.legal_actions())
        return self.step(action)

