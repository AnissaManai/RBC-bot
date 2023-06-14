# RBC-bot
The Source code for creating, training and testing a bot to play reconnaissance-blind-chess(RBC), a variant of chess with imperfect information. In this version a player cannot see the opponent's pieces but can reason about their location through sensing and other observations. For more information about RBC see [https://rbc.jhuapl.edu/](https://rbc.jhuapl.edu/) .

This work particularly focus on the sensing part of the game. 

A new RBC environment is developed to be used with the stable-baselines reinforcement learning algorithms. 
A model was first pretrained using supervised learning and some historical game data. Then it was trained further using a reinforcement learning algorithm called Proximal Policy Optimization (PPO) and self-play.
The neural network architecture used for the training is based on the Alpha-Zero architecture.
