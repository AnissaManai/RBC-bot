"""
This file is based on a0_resnet.py  from https://gitlab.com/jweil/PommerLearn

"""

import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish, Module, Dropout

from nn.builder_util import get_act, _Stem, _PolicyHead
from nn.RBCModel_SL import RBCModelSL



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")


class ResidualBlock(torch.nn.Module):
    """
    Definition of a residual block without any pooling operation
    """

    def __init__(self, channels, act_type, dropout):
        """
        :param channels: Number of channels used in the conv-operations
        :param act_type: Activation function to use
        """
        super(ResidualBlock, self).__init__()
        self.act_type = act_type

        self.body = Sequential(Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(num_features=channels),
                               Dropout(p=dropout),
                               get_act(act_type),
                               Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
                               BatchNorm2d(num_features=channels),
                               Dropout(p=dropout),
                               get_act(act_type))

    def forward(self, x):
        """
        Implementation of the forward pass of the residual block.
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Sum of the shortcut and the computed residual block computation
        """
        return x + self.body(x)


class AlphaZeroResnet(RBCModelSL):
    """ Creates the alpha zero net description based on the given parameters."""

    def __init__(
        self,
        n_labels=36,
        channels=256,
        nb_input_channels=21,
        board_height=8,
        board_width=8,
        channels_policy_head=2,
        num_res_blocks=19,
        act_type="relu",
        select_policy_from_plane=False,
        dropout = 0
    ):
        """
        :param n_labels: Number of labels the for the policy
        :param channels: Used for all convolution operations. (Except the last 2)
        :param channels_policy_head: Number of channels in the bottle neck for the policy head
        :param channels_value_head: Number of channels in the bottle neck for the value head
        :param num_res_blocks: Number of residual blocks to stack. In the paper they used 19 or 39 residual blocks
        :param value_fc_size: Fully Connected layer size. Used for the value output
        :return: net description
        """

        super(AlphaZeroResnet, self).__init__()
        self.nb_input_channels = nb_input_channels
        self.board_height = board_height
        self.board_width = board_width

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResidualBlock(channels, act_type, dropout=dropout))

        self.body = Sequential(_Stem(channels=channels, act_type=act_type,
                                     nb_input_channels=nb_input_channels, dropout=dropout),
                               *res_blocks)

        # create the heads which will be used in the fwd pass
        self.policy_head = _PolicyHead(board_height, board_width, channels, channels_policy_head, n_labels,
                                       act_type, select_policy_from_plane, dropout=dropout)

    def forward(self, flat_input):
        """
        Implementation of the forward pass of the full network
        Uses a broadcast add operation for the shortcut and the output of the residual block
        :param x: Input to the ResidualBlock
        :return: Policy Output
        """
        # input shape processing
        # x, state_bf = self.unflatten(flat_input)
        batch_size = flat_input.shape[0]
        x = flat_input.view(batch_size, self.nb_input_channels, self.board_height, self.board_width)

        out = self.body(x)

        policy = self.policy_head(out)

        return policy

    def get_state_shape(self, batch_size: int):
        raise NotImplementedError

