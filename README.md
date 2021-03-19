# KiK
Gra w kółko i krzyżyk na dużej planszy. Gracze na przemian stawiają swoje znaki. Wygrywa ten, kto pierwszy uzyska pięć znaków w rzędzie, kolumnie bądź skośnie.

Celem jest wytrenowanie sztucznej inteligencji za pomocą metod uczenia ze wzmocnieniem, która będzie ogrywała ludzi.

Język: Python

Biblioteki: tensorflow, numpy

Algorytm przeszukiwania drzewa: TODO

Metoda uczenia: policy learning

Architektura sieci: 

          tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),
          
          tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
          
          tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),
          tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
          # spłaszczamy sieć
          tf.keras.layers.Flatten(),
          # warstwa z maksymalną ilością połączeń
          tf.keras.layers.Dense(128, activation=tf.nn.relu),
          # ostatnia warstwa dająca prawdopodobieństwa wyboru poszczególnych pól na planszy
          tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax)

