#!/bin/python3.8

import torch
import torch.nn as nn

from src.ai.base_net import BaseNet, ArchitectureConfig
from src.ai.utils import mlp_creator
from src.core.data_types import Action
from src.core.logger import get_logger, cprint
from src.core.utils import get_filename_without_extension
from torchvision import transforms
import numpy as np
from PIL import Image
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
        self.output_size = (3,)
        self.discrete = False
        self.dropout = nn.Dropout(p=config.dropout) if config.dropout != 'default' else None
        self.encoder = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.decoder = mlp_creator(sizes=[8832, 2048, 256, self.output_size[0]],
                                   activation=nn.ReLU(),
                                   output_activation=nn.Identity(),
                                   bias_in_last_layer=False)
        self.initialize_architecture()


    def forward(self, inputs, train: bool = False) -> torch.Tensor:
        """
        Outputs steering action only
        """
        self.set_mode(train)
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        inputs = np.asarray(np.uint8(inputs[0]))
        inputs = inputs.transpose(2, 1, 0)
        inputs = Image.fromarray(inputs)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(inputs)
        input_batch = input_tensor.unsqueeze(0)
        if self._config.finetune:
            with torch.no_grad():
                x = self.encoder(input_batch)
        else:
            x = self.encoder(input_batch)
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

