# KiK

Value network is trained using policy improvement algorithm based on Bellman equation. Architecture of this network consists of two convolutional layers followed by two dense layers. Weights saved in the file "good_model.h5" are able to beat human occasionally.

Files:

agent - agent making decisions based on the observation

arena - testing network by playing against other networks

experience_buffer - collects data for training from simulations

kik_env - enviroment of the game

main - contains parameters and main runs

networks - class of the Value Network

player - pygame class for tasting against human

runner - runs experiments

training_algorithms - algorithms to process data for trainig
