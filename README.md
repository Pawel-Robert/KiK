# KiK
Gra w kółko i krzyżyk na dużej planszy. Gracze na przemian stawiają swoje znaki. Wygrywa ten, kto pierwszy uzyska pięć znaków w rzędzie, kolumnie bądź skośnie. Celem jest wytrenowanie sztucznej inteligencji za pomocą metod uczenia ze wzmocnieniem, która będzie ogrywała ludzi. Rozmiar planszy jest regulowany parametrami width oraz height (domyślnie 10 na 10). (Dalsze rozwinięcie gry może polegać na regulacji kształtu, którego narysowanie na planszy daje wygraną.)

Język: Python

Biblioteki: tensorflow, numpy

Algorytm przeszukiwania drzewa: TODO

Metoda uczenia: policy learning

W tracie uczenia dwóch graczy gra przeciwko sobie wykorzystując sieć do wyboru sowjego ruchu. Na podstawie wyborów gracza nr 1 (oraz jego ewentulanej wygranej/porażki) aktualizujemy parametry sieci w kroku treningowym. Jeden krok treningowy odpowiada jednej rozgrywce.

Architektura sieci: 

          Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),
          MaxPool2D(pool_size=(2, 2)),
          Flatten(),
          Dense(128, activation=tf.nn.relu),
          Dense(n_actions, activation=tf.nn.softmax)

