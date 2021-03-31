#!/usr/bin/python3
from typing import Tuple, Optional

from dataclasses import dataclass

import cv2
import torch
from torch.nn import *
import numpy as np
from dataclasses_json import dataclass_json
from tqdm import tqdm

from src.ai.base_net import BaseNet
from src.ai.utils import data_to_tensor
from src.ai.losses import *
from src.core.config_loader import Config
from src.core.data_types import Distribution
from src.core.logger import get_logger, cprint
from src.core.tensorboard_wrapper import TensorboardWrapper
from src.core.utils import get_filename_without_extension, create_output_video_segmentation_network
from src.data.data_loader import DataLoaderConfig, DataLoader

"""Given model, config, data_loader, evaluates a model and logs relevant training information

Depends on ai/architectures, data/data_loader, core/logger
"""


@dataclass_json
@dataclass
class EvaluatorConfig(Config):
    data_loader_config: DataLoaderConfig = None
    criterion: str = 'MSELoss'
    criterion_args_str: str = ''
    device: str = 'cpu'
    evaluate_extensive: bool = False
    store_output_on_tensorboard: bool = False
    store_feature_maps_on_tensorboard: bool = False
    store_projected_output_on_tensorboard: bool = False
    split_losses: bool = False

class Evaluator:

    def __init__(self, config: EvaluatorConfig, network: BaseNet, quiet: bool = False):
        self._config = config
        self._net = network
        self.data_loader = DataLoader(config=self._config.data_loader_config)

        if not quiet:
            self._logger = get_logger(name=get_filename_without_extension(__file__),
                                      output_path=config.output_path,
                                      quiet=False) if type(self) == Evaluator else None
            cprint(f'Started.', self._logger)

        self._device = torch.device(
            "cuda" if self._config.device in ['gpu', 'cuda'] and torch.cuda.is_available() else "cpu"
        )
        self._criterion = eval(f'{self._config.criterion}(reduction=\'none\', {self._config.criterion_args_str})')
        self._criterion.to(self._device)
        self._lowest_validation_loss = None
        self.data_loader.load_dataset()

        self._minimum_error = float(10**6)
        self._original_model_device = self._net.get_device() if self._net is not None else None

    def put_model_on_device(self, device: str = None):
        self._original_model_device = self._net.get_device()
        self._net.set_device(torch.device(self._config.device) if device is None else torch.device(device))

    def put_model_back_to_original_device(self):
        self._net.set_device(self._original_model_device)

    def evaluate(self, epoch: int = -1, writer=None, tag: str = 'validation') -> Tuple[str, bool]:
        self.put_model_on_device()
        total_error = []
        if self._config.split_losses:
            total_error_1 = []
            total_error_2 = []
        cntr = 0
#        for batch in tqdm(self.data_loader.get_data_batch(), ascii=True, desc='evaluate'):
        for batch in self.data_loader.get_data_batch():
            with torch.no_grad():
                predictions = self._net.forward(batch.observations, train=False)
                targets = data_to_tensor(batch.actions).type(self._net.dtype).to(self._device)
                error = self._criterion(predictions,
                                        targets).mean()
                if self._config.split_losses:
                    temp_loss = self._criterion(predictions, targets)
                    loss_1, loss_2 = torch.chunk(temp_loss, 2, dim=1)
                    loss_1 = loss_1.mean()
                    loss_2 = loss_2.mean()
                if self._config.split_losses:
                    total_error_1.append(loss_1.cpu().detach())
                    total_error_2.append(loss_2.cpu().detach())
                total_error.append(error)
                if self._config.store_projected_output_on_tensorboard:
                    numpy_obs = batch.observations[0].numpy()
                    zeros = np.zeros((1, 800, 848))
                    np_pred = predictions.cpu().numpy()
                    x_position = int(-np_pred[0][1] / np_pred[0][0] * 505.3 + 424.5)
                    y_position = int(-np_pred[0][2] / np_pred[0][0] * 505.3 + 400.5)
                    if 0<x_position<848 and 0<y_position<800:
                        cone_circle_cv = cv2.circle(numpy_obs[0, :, :], (x_position, y_position), int(3.65*np_pred[0][0]), 1, 5)
                        cone_circle_image_np = cone_circle_cv
                    else:
                        cone_circle_image_np = numpy_obs[0, :, :]
                    if self._config.split_losses:
                        x_position_1 = int(-np_pred[0][4] / np_pred[0][3] * 505.3 + 424.5)
                        y_position_1 = int(-np_pred[0][5] / np_pred[0][3] * 505.3 + 400.5)
                        if 0 < x_position_1 < 848 and 0 < y_position_1 < 800:
                            cone_circle_cv_1 = cv2.circle(cone_circle_image_np, (x_position_1, y_position_1),
                                                        int(3.65 * np_pred[0][3]), 1, 5)
                            cone_circle_image_np = np.asarray(cone_circle_cv_1)
                        else:
                            cone_circle_image_np = np.asarray(cone_circle_image_np)
                    zeros[0, :, :] = cone_circle_image_np
                    image_tensor = torch.from_numpy(zeros)
                    writer.write_output_image(image_tensor, f'{tag}/projected_cone/{cntr}')
                    cntr +=1
        error_distribution = Distribution(total_error)
        if self._config.split_losses:
            error_distribution_1 = Distribution(total_error_1)
            error_distribution_2 = Distribution(total_error_2)
        self.put_model_back_to_original_device()
        if writer is not None:
            writer.write_distribution(error_distribution, tag)
            if self._config.split_losses:
                writer.write_distribution(error_distribution_1, 'validation_cone_1')
                writer.write_distribution(error_distribution_2, 'validation_cone_2')
            if self._config.store_output_on_tensorboard and (epoch % 30 == 0 or tag == 'test'):
                writer.write_output_image(predictions, f'{tag}/predictions')
                writer.write_output_image(targets, f'{tag}/targets')
                writer.write_output_image(torch.stack(batch.observations), f'{tag}/inputs')

        if self._config.split_losses:
            msg = f' validation cone 1 {self._config.criterion} {error_distribution_1.mean: 0.3e} [{error_distribution_1.std:0.2e}] \n ' \
                  f' validation cone 2 {self._config.criterion} {error_distribution_2.mean: 0.3e} [{error_distribution_2.std:0.2e}]'
        else:
            msg = f' {tag} {self._config.criterion} {error_distribution.mean: 0.3e} [{error_distribution.std:0.2e}]'
        best_checkpoint = False
        if self._lowest_validation_loss is None or error_distribution.mean < self._lowest_validation_loss:
            self._lowest_validation_loss = error_distribution.mean
            best_checkpoint = True
        return msg, best_checkpoint

    def evaluate_extensive(self) -> None:
        """
        Extra offline evaluation methods for an extensive evaluation at the end of training
        :return: None
        """
        self.put_model_on_device('cpu')
        self.data_loader.get_dataset().subsample(10)
        dataset = self.data_loader.get_dataset()
        predictions = self._net.forward(dataset.observations, train=False).detach().cpu()
        #error = predictions - torch.stack(dataset.actions)
        self.put_model_back_to_original_device()

        # save_output_plots(output_dir=self._config.output_path,
        #                   data={'expert': np.stack(dataset.actions),
        #                         'network': predictions.numpy(),
        #                         'difference': error.numpy()})
        # create_output_video(output_dir=self._config.output_path,
        #                     observations=dataset.observations,
        #                     actions={'expert': np.stack(dataset.actions),
        #                              'network': predictions.numpy()})
        create_output_video_segmentation_network(output_dir=self._config.output_path,
                                                 observations=torch.stack(dataset.observations).numpy(),
                                                 predictions=predictions.numpy())

    def remove(self):
        self.data_loader.remove()
        [h.close() for h in self._logger.handlers]
