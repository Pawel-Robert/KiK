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
                    list.append(i*self.width+j)
        return list

    # def compute_coordinates(self, action):
    #     """ Recompute the coordinates on the board based on the action. """
    #
    #     return y_position, x_position

    def player_board(self, player):
        board = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i,j] == player:
                    board[i,j] = 1
        return board

    def inverse_board(self):
        self.board = - self.board


    def step(self, action):
        """" Main step function. Executes action and returns board, reward, done and info."""
        # print(f'action={action}')
        y_position = action // self.width
        x_position = action - y_position * self.width
        """ Change board at position action. """
        if self.board[y_position, x_position] == 0:
            self.board[y_position, x_position] = self.player
        else:
            print("BŁĄD!")

        self.reward = 0
        done = False

        """ Check winning condition. """
        if self.check_win(self.board):
            done = True
            self.reward = copy(self.player)

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
        for y in range(self.board.shape[0]):
            print('|', end='')
            for x in range(self.board.shape[1]):
                val = self.board[y][x]
                if val == 1.:
                    print(' o ', end='')
                elif val == -1.:
                    print(' x ', end='')
                else:
                    print(' . ', end='')
            print('|')
        # print(f'legal actions = {self.legal_actions()}')
        pass

    def q_render(self, network):
        """ Print the board. """
        for y in range(self.board.shape[0]):
            print('|', end='')
            for x in range(self.board.shape[1]):
                val = self.board[y][x]
                if val == 1.:
                    print(' o ', end='  ')
                elif val == -1.:
                    print(' x ', end='  ')
                else:
                    ac = np.zeros((self.height, self.width))
                    ac[y, x] = 1
                    ac_input = np.array([ac])
                    # st_1 = self.player_board(1)
                    # st_2 = self.player_board(-1)
                    st_input = np.array([self.board])
                    # st_input_2 = np.array([st_2])
                    current_q_value = float(network.model([st_input, ac_input])[0][0])
                    print(format(current_q_value,'.2f'), end=' ')
            print('|')
        # print(f'legal actions = {self.legal_actions()}')
        pass


    def check_if_in_range(self,a,b):
        return a in range(1, self.width+1) and b in range(1, self.height+1)


    def print_game_menu(self):
        """ Starting menu in the game """
        print('\n\nWITAMY W GRZE!\n')
        print('   |/ . |/ ')
        print('   |\ | |\ \n')
        print('Zasady gry: gracze na przemian stawiają swoje znaki (kółko lub krzyżyk) na planszy. Wygrywa gracz,')
        print('który jako pierwszy postawi pięć znaków w rzędzie, kolumnie bądź skośnie.')
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
            # print(type(action))
            return action


    def main_loop(self, mod, agent, network):
        """ Main gameplay loop."""
        while True:
            if not self.legal_actions():
                break
            print('Aktualny stan planszy:\n')
            self.q_render(network)
            if mod == '1':
                """ Take human action. """
                action = self.human_input()
            elif mod == '2':
                if self.player == -1:
                    """ Take human action. """
                    action = self.human_input()
                else:
                    """ Take AI action. """
                    action, q_value = agent.act(self.board, self.legal_actions(), 1)
                    print_q_value = format(q_value, '.2f')
                    print(f'Komputer wykonał ruch {action} o wartości {print_q_value}')
            elif mod == '3':
                board = self.player * self.board
                action, q_value = agent.act(board, self.legal_actions(), 1)
                print_q_value = format(q_value,'.2f')
                if self.player == 1:
                    print(f'Gracz komputerowy O wykonał ruch {action} o wartości {print_q_value}')
                else:
                    print(f'Gracz komputerowy X wykonał ruch {action} o wartości {print_q_value}')
                if action not in self.legal_actions():
                    for _ in range(1000):
                        print('Ruch niedozwolony!')
            else:
                action = np.random.choice(self.legal_actions())
                print(f'Komputer wykonał ruch {action}')

            """ Make a move. """
            state, reward, done, info = self.step(action)

            """ Ckech if someone already won. """
            if done:
                self.render()
                print(f'\n Wygrał gracz {-self.player}.')
                # time.sleep(2)
                break

    def game_play(self, agent, network):
        """" Game play! """
        self.print_game_menu()
        while True:
            print("Wybierz mod gry: HvH (1), HvAI (2), AIvAI (3), gracz losowy vs gracz losowy (4)")
            mod = input()
            self.reset()
            """ Main game loop. """
            self.main_loop(mod, agent, network)

            # print("Czy chesz zagrać jeszcze raz? (T/N)")
            #if input() == 'N':
            #    break


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


