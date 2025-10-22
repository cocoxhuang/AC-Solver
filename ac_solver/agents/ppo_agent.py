"""
This file contains Agent class that implements actor and critic networks.
"""

import math
import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn

def build_network(nodes_counts, std=0.01):
    """
    Constructs a neural network with fully connected layers and Tanh activations based on the specified node counts.

    Parameters:
    nodes_counts (list of int): A list where each element represents the number of nodes in a layer.
    std (float): The standard deviation for initializing the final layer's weights. Default is 0.01.

    Returns:
    list: A list of layers (including activation functions) representing the neural network.
    """
    layers = [nn.Linear(nodes_counts[0], nodes_counts[1]), nn.Tanh()]

    for i in range(1, len(nodes_counts) - 2):
        layers.append(nn.Linear(nodes_counts[i], nodes_counts[i + 1]))
        layers.append(nn.Tanh())

    layers.append(nn.Linear(nodes_counts[-2], nodes_counts[-1]))

    return layers

class AbPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, is_sinusoidal=False):
        super().__init__()
        self.is_sinusoidal = is_sinusoidal
        if is_sinusoidal:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        else:
            self.pe = nn.Embedding(max_len, d_model)
            self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        if self.is_sinusoidal:
            return x + self.pe[:x.size(1)]
        else:
            position_ids = self.position_ids[:x.size(1)]
            return x + self.pe(position_ids).unsqueeze(0)

def build_transformer(**kwargs):
    '''
    Builds a transformer-based policy network.
    TODO: dynamically increase vocab size via BPE
    '''

    vocab_size = kwargs.get('vocab_size', 5)
    max_token_len = kwargs.get('max_token_len', 512)
    d_model = kwargs.get('d_model', 64)
    n_layers = kwargs.get('n_layers', 3)
    n_heads = kwargs.get('n_heads', 4)
    activation = kwargs.get('activation', 'gelu')
    output_dim = kwargs.get('output_dim', 1)

    # 1. token embedding
    layers = [nn.Embedding(vocab_size, d_model)]

    # 2. positional encoding
    layers.append(AbPositionalEncoding(d_model, max_token_len))

    # transformer layers
    for _ in range(n_layers):
        layers.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                 batch_first=True, activation=activation))

    # output layer
    layers.append(nn.Linear(d_model, output_dim))

    return nn.Sequential(*layers)

class Agent(nn.Module):
    """
    A reinforcement learning agent that includes both a critic network for value estimation
    and an actor network for policy generation.

    Attributes:
    critic (nn.Sequential): The neural network used for value estimation.
    actor (nn.Sequential): The neural network used for policy generation.
    """

    def __init__(self, envs, nodes_counts):
        """
        Initializes the Agent with specified environment and node counts for the neural networks.

        Parameters:
        envs (gym.Env): The environment for which the agent is being created.
        nodes_counts (list of int): A list where each element represents the number of nodes in a hidden layer.
        """
        super(Agent, self).__init__()

        input_dim = np.prod(envs.single_observation_space.shape)

        # self.critic_nodes = [input_dim] + nodes_counts + [1]
        # self.actor_nodes = [input_dim] + nodes_counts + [envs.single_action_space.n]

        # self.critic = nn.Sequential(*build_network(self.critic_nodes, 1.0))
        # self.actor = nn.Sequential(*build_network(self.actor_nodes, 0.01))
        
        critic_args = {
            'vocab_size': 5, # {-2, -1, 0, 1, 2}
            'max_token_len': input_dim,
            'd_model': 512,
            'n_layers': 1,
            'n_heads': 1,
            'activation': 'gelu',
            'output_dim': 1
        }
        actor_args = {
            'vocab_size': 5, # {-2, -1, 0, 1, 2}
            'max_token_len': input_dim,
            'd_model': 512,
            'n_layers': 1,
            'n_heads': 1,
            'activation': 'gelu',
            'output_dim': envs.single_action_space.n
        }

        self.critic = build_transformer(**critic_args)
        self.critic.apply(self.initialize_layers)

        self.actor = build_transformer(**actor_args)
        self.actor.apply(self.initialize_layers)

    def get_value(self, x):
        """
        Computes the value of a given state using the critic network.

        Parameters:
        x (torch.Tensor): The input tensor representing the state.

        Returns:
        torch.Tensor: The value of the given state.
        """

        # a naive tokenizer:
        # add 2 to make all values non-negative
        x = (x + 2).long()

        # return self.critic(x)
        return self.critic(x)[:, -1, :]

    def get_action_and_value(self, x, action=None):
        """
        Computes the action to take and its associated value, log probability, and entropy.

        Parameters:
        x (torch.Tensor): The input tensor representing the state.
        action (torch.Tensor, optional): The action to evaluate. If None, a new action will be sampled.

        Returns:
        tuple: A tuple containing the action, its log probability, the entropy of the action distribution, and the value of the state.
        """
        # a naive tokenizer:
        # add 2 to make all values non-negative
        x = (x + 2).long()

        # logits = self.actor(x)
        # value = self.critic(x)
        logits = self.actor(x)[:, -1, :]  # (B, n_actions)
        value = self.critic(x)[:, -1, :]
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value
    
    def initialize_layers(self, layer, std=0.02, bias_const=0.0):
        """
        Initializes the weights and biases of all layers.

        Parameters:
        layer (nn.Module): The neural network layer to initialize. 
        std (float): The standard deviation for orthogonal initialization of weights. Default is sqrt(2).
        bias_const (float): The constant value to initialize the biases. Default is 0.0.

        Returns:
        nn.Module: The initialized layer.
        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
        elif isinstance(layer, nn.Embedding):
            torch.nn.init.orthogonal_(layer.weight, std)        