import logging
from typing import Type, Any, Callable, Union, List, Optional


import torch
import torch.nn as nn
import torchvision.models as M
from torchvision.models.vision_transformer import _vision_transformer


class MLPHead(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """
    Fully connected network.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: tuple,
        activation: str = "gelu",
        normalize_input=False,
    ):
        super().__init__()

        layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.activation = activation

        # Normalize input
        if normalize_input:
            layers.append(nn.BatchNorm1d(in_channels))

        # First layer
        layers.append(nn.Linear(in_channels, hidden_channels[0]))
        layers.append(nn.BatchNorm1d(hidden_channels[0]))
        layers.append(self._activation(activation))

        self.input_layer = torch.nn.Sequential(*layers)

        # Hidden layers
        if len(hidden_channels) > 0:
            self.hidden = nn.ModuleList()
            for i in range(len(hidden_channels) - 1):
                layers = []
                layers.append(nn.Linear(hidden_channels[i], hidden_channels[i + 1]))
                layers.append(nn.BatchNorm1d(hidden_channels[i + 1]))
                layers.append(self._activation(activation))
                self.hidden.append(torch.nn.Sequential(*layers))

        self.output_layer = nn.Linear(hidden_channels[-1], out_channels)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden:
            x = layer(x)

        x = self.output_layer(x)
        return x

    @staticmethod
    def _activation(activation: str) -> Callable:
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")()


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, batchnorm=True):
        super().__init__()
        if batchnorm:
            self.linear = nn.Sequential(
                nn.BatchNorm1d(input_dim), torch.nn.Linear(input_dim, output_dim)
            )
        else:
            self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x


class Classification_Head(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super().__init__()

        layers = [nn.Batchnorm1d(in_channels)] + [] * n_hidden + [nn.Linear(mlp_hidden_size)]

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


def _get_transformer(config):
    return _vision_transformer(
        patch_size=config["model"]["patch_size"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        hidden_dim=config["model"]["features"],
        mlp_dim=config["model"]["mlp_dim"],
        weights=None,
        progress=True,
    )


def _get_net(config):
    networks = {
        "resnet18": M.resnet18,
        "resnet34": M.resnet34,
        "resnet50": M.resnet50,
        "resnet101": M.resnet101,
        "resnet152": M.resnet152,
        "wide_resnet50_2": M.wide_resnet50_2,
        "wide_resnet101_2": M.wide_resnet101_2,
        "efficientnetb7": M.efficientnet_b7,
        "efficientnetb0": M.efficientnet_b0,  # not tested, could be v useful re zoobot
    }

    return networks[config["model"]["architecture"]]()
