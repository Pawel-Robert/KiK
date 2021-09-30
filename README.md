# KiK

The purpose of this project is to train a neural network, which will be able to compete with humans in the game of Gomoku. This is a Japanese game, which is similar to tic-tac-toe, however player on a big board (for example 13x13 spaces) and to win the game it is necessary to get 5 marks in a line. 

Version "ValueNetwork" is the latest one. It uses value network together trained using Bellman equation and is able to beat human occasionally. In the file "good_model.h5" are saved weigths of the of the trained models.

Version is in the branch "Target Network" it is special case of 3x3 board (tic-tac-toe). It uses Q network trained using Bellman equation.

Next for the near future aim is to use Monte Carlo Tree Search algorithm. 

Language: Python

Libraries: tensorflow, numpy.
