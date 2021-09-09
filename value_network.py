import tensorflow as tf

class ValueNetwork:
    def __init__(self, width, height, model_path = None):
        self.width = width
        self.height = height
        self.n_actions = width*height
        if model_path is not None:
           self.model = model_path
        else:
            model = tf.keras.models.Sequential([
                # na początku tworzymy kilka warstw konwolucyjnych przetwarzających planszę
                tf.keras.layers.Conv2D(filters=26, kernel_size=(3, 3), activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                # spłaszczamy sieć
                tf.keras.layers.Flatten(),
                # warstwa z maksymalną ilością połączeń
                tf.keras.layers.Dense(128, activation=tf.nn.relu),
                # ostatnia warstwa dająca prawdopodobieństwa wyboru poszczególnych pól na planszy
                tf.keras.layers.Dense(self.n_actions, activation=tf.nn.softmax)
            ])
            self.model = model

    def predict_value(self, state):
        return self.model.predict(state)


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
