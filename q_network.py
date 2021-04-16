from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Concatenate
from keras.models import Model


class QNetworkLarger:
    def __init__(self, width, height):
        input_state = Input(shape=(height, width, 1))
        input_action = Input(shape=(height, width, 1))
        state =  Conv2D(filters=26, kernel_size=(3, 3), activation='relu', padding='same')(input_state)
        state = MaxPool2D(pool_size=(2, 2))(state)
        state = Conv2D(filters=36, kernel_size=(3, 3), activation='relu', padding='same')(state)
        state = MaxPool2D(pool_size=(2, 2),)(state)

        action = Conv2D(filters=26, kernel_size=(3, 3), activation='relu', padding='same')(input_action)
        action = MaxPool2D(pool_size=(2, 2))(action)
        action = Conv2D(filters=36, kernel_size=(2, 2), activation='relu', padding='same')(action)
        action = MaxPool2D(pool_size=(2, 2))(action)

        state_and_action = Concatenate()([state, action])
        state_and_action = Conv2D(filters=26, kernel_size=(3, 3), activation='relu', padding='same')(state_and_action)
        state_and_action = MaxPool2D(pool_size=(2, 2))(state_and_action)

        # spłaszczamy sieć
        q_value = Flatten()(state_and_action)
        # warstwa z maksymalną ilością połączeń
        q_value = Dense(28, activation='relu')(q_value)
        # ostatnia warstwa dająca prawdopodobieństwa wyboru poszczególnych pól na planszy

        q_value = Dense(28)(q_value)

        self.model = Model(inputs=[input_state, input_action], outputs=[q_value])
        self.model.compile(loss='mse', metrics=['loss', 'accuracy'])



class QNetwork3x3:
    def __init__(self):
        input_state = Input(shape=(9,))
        input_action = Input(shape=(9,))
        state =  Dense(50, activation='relu')(input_state)
        action = Dense(50, activation='relu')(input_action)
        state_and_action = Concatenate()([state, action])

        q_value = Dense(50, activation='relu')(state_and_action)
        # warstwa z maksymalną ilością połączeń
        q_value = Dense(50, activation='relu')(q_value)
        # ostatnia warstwa dająca prawdopodobieństwa wyboru poszczególnych pól na planszy
        q_value = Dense(1)(q_value)
        self.model = Model(inputs=[input_state, input_action], outputs=[q_value])
        self.model.compile(loss='mse', metrics=['mse'])#, optimizer=keras.optimizers.Adam(1r=0.03))



 # JAK UZYC TEJ SIECI:
# import numpy as np
# st = np.zeros(9)
# st[0] = 1.
# st_input = np.array([st])
# ac = np.zeros(9)
# ac[2] = -1.
# ac_input = np.array([ac])
# d = QNetwork3x3()
# print(d.model.predict([st_input,ac_input]))