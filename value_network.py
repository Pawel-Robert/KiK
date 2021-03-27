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
                tf.keras.layers.Conv2D(filters=26, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=(height, width)),
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
