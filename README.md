# KiK

This is a personal project in reinforcement learning. Language: Python. Libraries: tensorflow, numpy.

The purpose of this project is to train a neural network, which will be able to compete with humans in the game of Gomoku. This is a Japanese game, which is similar to tic-tac-toe, however player on a big board (for example 13x13 spaces) and to win the game it is necessary to get 5 marks in a line. The environement of the game is implemented in the file "kik_env.py" (acronym "KiK" comes from "kółko i krzyżyk", which means tic-tac-toe in Polish). This environement class is compatible with the OpenAI specification (it contains functions: step, render, reset).

Version "ValueNetwork" is the latest one. As the name is suggesting the network predicts value function (i.e. expected reward for a game starting in a given state). Network is trained using Bellman equation. In the file "good_model.h5" are saved weigths of the of the trained models, using which network is able to beat human occasionally.

Version is in the branch "Target Network" it is special case of 3x3 board (tic-tac-toe). It uses Q network trained using Bellman equation.

Next aim for the near future is to use Monte Carlo Tree Search algorithm. 

Language: Python

Libraries: tensorflow, numpy.
