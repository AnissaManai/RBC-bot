
import gym
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, LeakyReLU, Sigmoid, Tanh, Linear, Hardsigmoid, Hardswish, Module, Dropout
from email import policy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from stable_baselines3.common.vec_env import dummy_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from nn.RBCModel import RBCModel

from nn.a0_resnet import AlphaZeroResnet
from nn.builder_util import get_act, _Stem, _PolicyHead, _ValueHead

from gym_env import RBCEnv
from gym.spaces import Space, Box



class ResidualBlock(th.nn.Module):
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

class AlphaZeroResnet(nn.Module):
    """ :param observation_space: (gym.Space) :param features_dim: (int) Number of features extracted. This corresponds to the number of unit for the last layer. """

    def __init__(
        self, 
        observation_space: Box, 
        channels = 256, 
        board_height=8,
        board_width=8,
        channels_value_head=1,
        channels_policy_head=2,
        num_res_blocks=19,
        value_fc_size=256,
        act_type="relu",
        select_policy_from_plane=False,
        dropout = 0, 
        last_layer_dim_pi: int = 36, 
        last_layer_dim_vf: int = 1
        
    ):
        super(AlphaZeroResnet, self).__init__()
        self.nb_input_channels = observation_space.shape[0]
        self.board_height = board_height
        self.board_width = board_width

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResidualBlock(channels, act_type, dropout=dropout))

        self.body = Sequential(_Stem(channels=channels, act_type=act_type,
                                     nb_input_channels=self.nb_input_channels, dropout=dropout),
                               *res_blocks)

        # create the heads which will be used in the fwd pass
        
        self.policy_net = _PolicyHead(board_height, board_width, channels, channels_policy_head, last_layer_dim_pi,
                                       act_type, select_policy_from_plane, dropout=dropout)

        self.value_net = _ValueHead(board_height, board_width, channels, channels_value_head, value_fc_size, act_type)

    def forward(self, flat_input) -> Tuple[th.Tensor, th.Tensor]:
        # input shape processing
        # x, state_bf = self.unflatten(flat_input)
        batch_size = flat_input.shape[0]
        x = flat_input.view(batch_size, self.nb_input_channels, self.board_height, self.board_width)

        out = self.body(x)
        # print('out shape ', out.shape)
        policy = self.policy_net(out)
        value = self.value_net(out)

        
        # print('policy shape ', policy.shape)
        # print('value shape ', value.shape)
        

        return  policy, value

    def forward_actor(self, flat_input) -> th.Tensor:
        # # input shape processing
        # # x, state_bf = self.unflatten(flat_input)
        batch_size = flat_input.shape[0]
        # print('forward actor flat input shape  ', flat_input.shape)
        x = flat_input.view(batch_size, self.nb_input_channels, self.board_height, self.board_width)
        # print('forward actor x shape  ', x.shape)
        out = self.body(x)
        # print('forward actor out shape  ', out.shape)
        
        policy = self.policy_net(out)
        return policy
        # return self.policy_net(flat_input)

    def forward_critic(self, flat_input) -> th.Tensor: 
        # # input shape processing
        # # x, state_bf = self.unflatten(flat_input)
        batch_size = flat_input.shape[0]
        # print('forward critic flat input shape  ', flat_input.shape)
        x = flat_input.view(batch_size, self.nb_input_channels, self.board_height, self.board_width)
        # print('forward critic x shape  ', x.shape)
        out = self.body(x)
        # print('forward actor out shape  ', out.shape)
        value = self.value_net(out)
        return value
        # return self.value_net(flat_input)


class CustomActorCriticPolicy(ActorCriticPolicy, RBCModel): 
    def __init__(
        self,
        observation_space: Space, 
        action_space: Space, 
        lr_schedule: Callable[[float], float],
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,

    ):

        super(CustomActorCriticPolicy, self).__init__(
                observation_space,
                action_space,
                lr_schedule,
                activation_fn,
                # Pass remaining arguments to base class
                *args,
                **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False
        self.observation_space = observation_space
        self.action_space = action_space

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AlphaZeroResnet(self.observation_space, dropout=0.2)