#!/bin/python3.8

import torch
import torch.nn as nn

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension

"""
Tiny four encoding and three decoding layers with dropout.
Expects 3x200x200 inputs and outputs 3c 
"""


class Net(BaseNet):

    def __init__(self, config: ArchitectureConfig, quiet: bool = False):
        super().__init__(config=config, quiet=True)
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quiet=False)
        if not quiet:
            cprint(f'Started.', self._logger)
        self.input_size = (3, 800, 848)
        self.output_size = (4,)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout != 'default' else None
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=3),
            nn.Conv2d(4, 8, 5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=3),
            nn.Conv2d(8, 16, 5, stride=2),
            nn.ReLU(),
        )
        self.decoder = mlp_creator(sizes=[1296, 512, 128, self.output_size[0]],
                                   activation=nn.ReLU(),
                                   output_activation=nn.Identity(),
                                   bias_in_last_layer=False)
        self.initialize_architecture()


    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        self.set_mode(train)
        inputs = self.process_inputs(inputs=inputs)
        if self._config.finetune:
            with torch.no_grad():
                x = self.encoder(inputs)
        else:
            x = self.encoder(inputs)
        x = x.flatten(start_dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.decoder(x)
        return x

    def get_action(self, inputs, train: bool = False) -> Action:
        inputs = self.process_inputs(inputs=inputs)
        output = self.forward(inputs)
        return Action(actor_name=get_filename_without_extension(__file__),
                      value=output.data)

