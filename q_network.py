from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Concatenate
from keras.models import Model


class QNetwork3x3:
    def __init__(self):
        input_state_1 = Input(shape=(9,))
        input_state_2 = Input(shape=(9,))
        input_action = Input(shape=(9,))
        state_1 =  Dense(50, activation='relu')(input_state_1)
        state_2 =  Dense(50, activation='relu')(input_state_2)
        action = Dense(5, activation='relu')(input_action)
        state_and_action = Concatenate()([state_1, state_2, action])

        q_value = Dense(50, activation='relu')(state_and_action)
        # warstwa z maksymalną ilością połączeń
        q_value = Dense(50, activation='relu')(q_value)
        # q_value = Dense(50, activation='relu')(q_value)
        # ostatnia warstwa dająca prawdopodobieństwa wyboru poszczególnych pól na planszy
        q_value = Dense(1)(q_value)
        self.model = Model(inputs=[input_state_1, input_state_2, input_action], outputs=[q_value])
        self.model.compile(loss='mse', metrics=['mse'])#, optimizer=keras.optimizers.Adam(lr=0.03))



 # JAK UZYC TEJ SIECI:lass QNetwork3x3:
# import numpy as np
# st = np.zeros(9)
# st[0] = 1.
# st_input = np.array([st])
# ac = np.zeros(9)
# ac[2] = -1.
# ac_input = np.array([ac])
# d = QNetwork3x3()
# print(d.model.predict([st_input,ac_input]))