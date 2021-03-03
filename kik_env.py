import numpy as np

class KiKEnv():
    #metadata = {'render.modes': ['human']}

    #inicjując podajemy rozmiary planszy.
    def __init__(self,wysokosc,szerokosc, player = 1):

        # plasza o rozmiarach wysokoscx szerokosc na której stawiamy na przemian znaki
        plansza = np.zeros((wysokosc,szerokosc))
        pass

    #spawdzamy, czy na plaszy pojawił się kształt dający zwycięstwo
    def check_win(self):
        for i in range(wysokosc-5):
            for j in range(szerokosc-5):
                if compute_filter(plansza,a,b):
                    return True
        return False



    # sprawdzamy zwycięstwo lokalnie w kwadracie 5x5 o górnym lewym wierzchołku w pozycji (a,b)
    def compute_filter(plansza,a,b):
        poziomo = 0
        pionowo = 0
        skosnie = 0
        antyskosnie = 0
        for n in range(5):
            poziomo = poziomo + plansza[a+n,b]
            pionowo = pionowo + plansza[a,b+n]
            skosnie = skosnie + plansza[a+n,b+n]
            antyskosnie = antyskosnie + plansza[a+4-n,b+n]
        if max(poziomo, pionowo, skosnie, antyskosnie) == 5:
            self.reward = 1
            return True
        elif min(poziomo, pionowo, skosnie, antyskosnie) == -5:
            self.reward = 0
            return True
        return False

    # funkcja spawdzająca, czy wykoanie danej akcji jest poprawne, to znaczy czy pole nie jest już zajęte
    def is_allowed_move(self,a,b):
        if plansza[a,b] == 0:
            return True
        else:
            return False

    # funckja step
    def step(self, action):
        plansza(action) = player
        if self.check_win():
            done = True
            reward = self.reward
        return plansza, reward, done

    def reset(self):
        plansza = np.zeros((wysokosc,szerokosc))
        pass

    def render(self, mode='human', close=False):
        print(plansza)
        pass