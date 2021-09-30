# KiK

The purpose of this project is to train a neural network, which will be able to compete with humans in the game of Gomoku. This is a Japanese game, which is similar to tic-tac-toe, however player on a big board (typically 15x15 spaces). Reinforcement learning algorithm, which is going to be used in this purpose is Monte Carlo Tree Search, implemented as in the AlphaGo project. 

Version "ValueNetwork" is the latest one. It uses value network together trained using Bellman equation and is able to beat human occasionally. In the file "good_model.h5" are saved weigths of the of the trained models.

Version is in the branch "Target Network" it is special case of 3x3 board (tic-tac-toe). It uses Q network trained using Bellman equation.


Language: Python

Libraries: tensorflow, numpy.
