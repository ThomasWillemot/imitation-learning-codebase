#!/usr/bin/python3.7
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import List

import torch
from dataclasses_json import dataclass_json
from torch import nn

from src.ai.architectures import *  # Do not remove
from src.core.config_loader import Config
from src.core.logger import get_logger, cprint, MessageType
from src.core.utils import get_filename_without_extension

"""
Model contains of architectures (can be modular).
Model ensures proper initialization and storage of model parameters.
"""


class InitializationType(IntEnum):
    Xavier = 0
    Constant = 1


@dataclass_json
@dataclass
class ModelConfig(Config):
    device: str = 'cpu'
    load_checkpoint_dir: str = None
    architecture: str = None
    dropout: float = 0.
    output_sizes: List[List] = None
    initialisation_type: InitializationType = InitializationType.Xavier
    pretrained: bool = False
    initialisation_seed: int = 0

    def __post_init__(self):
        if self.load_checkpoint_dir is None:
            del self.load_checkpoint_dir
        if self.output_sizes is None:
            self.output_sizes = [[6]]  # default to six dimensional action space.


class Model:

    def __init__(self, config: ModelConfig):
        self._config = config
        self._logger = get_logger(name=get_filename_without_extension(__file__),
                                  output_path=config.output_path,
                                  quite=False)
        cprint(f'Started.', self._logger)
        self._checkpoint_directory = os.path.join(self._config.output_path, 'torch_checkpoints')
        os.makedirs(self._checkpoint_directory, exist_ok=True)
        self._architecture = eval(f'{self._config.architecture}.Net('
                                  f'dropout={self._config.dropout},'
                                  f'output_sizes={self._config.output_sizes})')

        if self._config.load_checkpoint_dir:
            self._config.load_checkpoint_dir = self._config.load_checkpoint_dir \
                if self._config.load_checkpoint_dir.endswith('torch_checkpoints') \
                else os.path.join(self._config.load_checkpoint_dir, 'torch_checkpoints')
            self.load_from_checkpoint(checkpoint_dir=self._config.load_checkpoint_dir)
        else:
            self.initialize_architecture_weights(self._config.initialisation_type)

        self.device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self._architecture.to(self.device)

    def forward(self, inputs: List[torch.Tensor], train: bool = False):
        inputs = [i.to(self.device) for i in inputs]
        return self._architecture.forward(inputs=inputs, train=train)

    def initialize_architecture_weights(self, initialisation_type: InitializationType = 0):
        torch.manual_seed(self._config.initialisation_seed)
        for p in self.get_parameters():
            if initialisation_type == InitializationType.Xavier:
                if len(p.shape) == 1:
                    nn.init.uniform_(p, a=0, b=1)
                else:
                    nn.init.xavier_uniform_(p)
            elif initialisation_type == InitializationType.Constant:
                nn.init.constant_(p, 0.001)
            else:
                raise NotImplementedError

    def load_from_checkpoint(self, checkpoint_dir: str):
        if len(os.listdir(checkpoint_dir)) == 0:
            cprint(f'Could not find suitable checkpoint in {checkpoint_dir}', self._logger, MessageType.error)
            raise FileNotFoundError
        if os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_best')):
            self._architecture.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_best')))
        elif os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_latest')):
            self._architecture.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_latest')))
        else:
            checkpoints = {int(f.split('_')[-1]):
                           os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)}
            latest_checkpoint = checkpoints[max(checkpoints.keys())]
            self._architecture.load_state_dict(torch.load(latest_checkpoint))

    def save_to_checkpoint(self, tag: str = ''):
        filename = f'checkpoint_{tag}' if tag != '' else 'checkpoint'
        torch.save(self._architecture.state_dict(), f'{self._checkpoint_directory}/{filename}')
        torch.save(self._architecture.state_dict(), f'{self._checkpoint_directory}/checkpoint_latest')
        cprint(f'stored {filename}', self._logger)

    def get_input_sizes(self):
        return self._architecture.input_sizes

    def get_output_sizes(self):
        return self._architecture.output_sizes

    def get_parameters(self) -> list:
        return list(self._architecture.parameters())