import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """Wrapper for setting limits for parameter initialization"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class QNetwork(nn.Module):
    """
    Deep Q-Learning network architecture in PyTorch
    """

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, gate=F.relu):

        super(QNetwork, self).__init__()
        self.gate = gate

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):

        x = self.gate(self.fc1(state))
        x = self.gate(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """Wrapper for PyTorch neural network template for Actor in deep reinforcement learning algorithms"""
    def __init__(self,
                 state_size,
                 action_size,
                 fc1_units=400,
                 fc2_units=300,
                 gate=F.relu,
                 batch_normalize=True):
        """
        :param state_size (int): Input data size
        :param action_size (int): Output data size
        :param fc1_units (int): Number of nodes in first layer
        :param fc2_units (int): Number of nodes in second layer
        :param gate: Torch activation function
        :param batch_normalize (bool): Whether to batch normalize or not
        """

        super(Actor, self).__init__()
        self.gate = gate

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # Batch normalization
        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.bn1 = nn.BatchNorm1d(fc1_units)

        self.reset_parameters()

    def reset_parameters(self):
        """Wrapper to reset and reinitialize parameters in all layers"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = self.gate(self.fc1(state))
        if self.batch_normalize:
            x = self.bn1(x)
        x = self.gate(self.fc2(x))
        # tanh activation for action space in range -1,1
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Wrapper for PyTorch neural network template for Actor in deep reinforcement learning algorithms"""
    def __init__(self,
                 state_size,
                 action_size,
                 fc1_units=400,
                 fc2_units=300,
                 gate=F.relu,
                 batch_normalize=True,
                 multi_critic=False):
        """
        :param state_size (int): Input data size
        :param action_size (int): Output data size
        :param fc1_units (int): Number of nodes in first layer
        :param fc2_units (int): Number of nodes in second layer
        :param gate: Torch activation function
        :param batch_normalize (bool): Whether to batch normalize or not. Default True
        :param multi_critic (bool): True if multiple interacting agents, False if not. Default False
        """

        super(Critic, self).__init__()
        self.multi_critic = multi_critic
        self.gate = gate
        # Concatenating action space into one of the layers of the network happens at different layers
        # Depending if the agents interact with each other or not
        if self.multi_critic:
            # Concat state and action space on first layer for multiple interacting agent
            self.fc1 = nn.Linear(state_size + action_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, 1)

        else:
            self.fc1 = nn.Linear(state_size, fc1_units)
            # Concat state and action space on layer before output layer
            self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
            self.fc3 = nn.Linear(fc2_units, 1)

        # batch normalization
        self.batch_normalize = batch_normalize
        if self.batch_normalize:
            self.bn1 = nn.BatchNorm1d(fc1_units)

        self.reset_parameters()

    def reset_parameters(self):
        """Wrapper to reset and reinitialize parameters in all layers"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build an critic (value) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # Concatenating action space into one of the layers of the network happens at different layers
        # Depending if the agents interact with each other or not
        if self.multi_critic:
            # Concat state and action space on first layer for multiple interacting agent
            x = torch.cat((state, action.float()), dim=1)
            x = self.gate(self.fc1(x))
            if self.batch_normalize:
                x = self.bn1(x)
            x = self.gate(self.fc2(x))
        else:
            x = self.gate(self.fc1(state))
            if self.batch_normalize:
                x = self.bn1(x)
            x = torch.cat((x, action.float()), dim=1)
            x = self.gate(self.fc2(x))

        return self.fc3(x)

