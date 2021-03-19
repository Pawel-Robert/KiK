# KiK
Gra w kółko i krzyżyk na dużej planszy. Gracze na przemian stawiają swoje znaki. Wygrywa ten, kto pierwszy uzyska pięć znaków w rzędzie, kolumnie bądź skośnie.

Celem jest wytrenowanie sztucznej inteligencji za pomocą metod uczenia ze wzmocnieniem, która będzie ogrywała ludzi.

Język: Python

Biblioteki: tensorflow, numpy

Algorytm przeszukiwania drzewa: TODO

Metoda uczenia: policy learning

Architektura sieci: 

          Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),
          MaxPool2D(pool_size=(2, 2)),
          Flatten(),
          Dense(128, activation=tf.nn.relu),
          Dense(n_actions, activation=tf.nn.softmax)

