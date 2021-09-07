from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Concatenate
from keras.models import Model
import numpy as np

class QValue:
    def __init__(self):
        input_state = Input(shape=(9,))
        input_action = Input(shape=(9,))

        state_and_action = Concatenate()([input_state, input_action])
        layer = Dense(36, activation='relu')(state_and_action)
        layer = Dense(36, activation='relu')(layer)
        q_value = Dense(1)(layer)

        self.model = Model(inputs=[input_state, input_action], outputs=[q_value])
        self.model.compile(loss='mse', metrics=['mse'])

    def evaluate(self, state, action):
        return self.model([np.array([state.flatten()]), np.array([action])])[0][0]

# class QNetwork3x3:
#     def __init__(self):
#         input_state_1 = Input(shape=(9,))
#         input_state_2 = Input(shape=(9,))
#         input_action = Input(shape=(9,))
#         state_1 = Dense(200, activation='relu')(input_state_1)
#         state_2 = Dense(200, activation='relu')(input_state_2)
#         action = Dense(200, activation='relu')(input_action)
#         state_and_action = Concatenate()([state_1, state_2, action])
#
#         q_value = Dense(200, activation='relu')(state_and_action)
#         q_value = Dense(200, activation='relu')(q_value)
#         q_value = Dense(200, activation='relu')(q_value)
#         q_value = Dense(1)(q_value)
#         self.model = Model(inputs=[input_state_1, input_state_2, input_action], outputs=[q_value])
#         self.model.compile(loss='mse', metrics=['mse'])

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


class ValueNetwork3x3:
    def __init__(self):
        input_state_1 = Input(shape=(9,))
        input_state_2 = Input(shape=(9,))
        state_1 =  Dense(50, activation='relu')(input_state_1)
        state_2 =  Dense(50, activation='relu')(input_state_2)
        state = Concatenate()([state_1, state_2])

        value = Dense(50, activation='relu')(state)
        value = Dense(50, activation='relu')(value)
        value = Dense(50, activation='relu')(value)
        value = Dense(1)(value)
        self.model = Model(inputs=[input_state_1, input_state_2], outputs=[value])
        self.model.compile(loss='mse', metrics=['mse'])



