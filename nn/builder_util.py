"""
Source: https://gitlab.com/jweil/PommerLearn

Utility methods for building the neural network architectures.
"""

import math
import torch
from torch.nn import Sequential, Conv1d, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish,\
    Module, AdaptiveAvgPool2d, BatchNorm1d, Dropout

def get_act(act_type):
    """Wrapper method for different non linear activation functions"""
    if act_type == "relu":
        return ReLU()
    if act_type == "sigmoid":
        return Sigmoid()
    if act_type == "tanh":
        return Tanh()
    if act_type == "lrelu":
        return LeakyReLU(negative_slope=0.2)
    if act_type == "hard_sigmoid":
        return Hardsigmoid()
    if act_type == "hard_swish":
        return Hardswish()
    raise NotImplementedError


class _Stem(torch.nn.Module):
    def __init__(self, channels, act_type="relu", nb_input_channels=34):
        """
        Definition of the stem proposed by the alpha zero authors
        :param channels: Number of channels for 1st conv operation
        :param act_type: Activation type to use
        :param nb_input_channels: Number of input channels of the board representation
        """

        super(_Stem, self).__init__()

        self.body = Sequential(
            Conv2d(in_channels=nb_input_channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1),
                   bias=False),
            BatchNorm2d(num_features=channels),
            Dropout(p=0.2),
            get_act(act_type))

    def forward(self, x):
        """
        Compute forward pass
        :param F: Handle
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        return self.body(x)


class _PolicyHead(Module):
    def __init__(self, board_height=11, board_width=11, channels=256, policy_channels=2, n_labels=4992, act_type="relu",
                 select_policy_from_plane=False):
        """
        Definition of the policy head proposed by the alpha zero authors
        :param policy_channels: Number of channels for 1st conv operation in branch 0
        :param act_type: Activation type to use
        channelwise squeeze excitation, channel-spatial-squeeze-excitation, respectively
        """

        super(_PolicyHead, self).__init__()

        self.body = Sequential()
        self.select_policy_from_plane = select_policy_from_plane

        if self.select_policy_from_plane:
            self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels, padding=1, kernel_size=(3, 3), bias=False),
                                   BatchNorm2d(num_features=channels),
                                   Dropout(p=0.2),
                                   get_act(act_type),
                                   Conv2d(in_channels=channels, out_channels=policy_channels, padding=1, kernel_size=(3, 3), bias=False))
            self.nb_flatten = policy_channels*board_width*policy_channels

        else:
            self.body = Sequential(Conv2d(in_channels=channels, out_channels=policy_channels, kernel_size=(1, 1), bias=False),
                                   BatchNorm2d(num_features=policy_channels),
                                   Dropout(p=0.2),
                                   get_act(act_type))

            self.nb_flatten = board_height*board_width*policy_channels
            self.body2 = Sequential(Linear(in_features=self.nb_flatten, out_features=n_labels))

    def forward(self, x):
        """
        Compute forward pass
        :param x: Input data to the block
        :return: Activation maps of the block
        """
        if self.select_policy_from_plane:
            return self.body(x).view(-1, self.nb_flatten)
        else:
            x = self.body(x).view(-1, self.nb_flatten)
            return self.body2(x)
