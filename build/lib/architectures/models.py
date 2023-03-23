import logging
from typing import Type, Any, Callable, Union, List, Optional

from resnet import _get_resnet

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


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, batchnorm=False):
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
