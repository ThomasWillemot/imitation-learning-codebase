import os
import shutil
import unittest

import torch

from src.ai.base_net import InitializationType, ArchitectureConfig, BaseNet
from src.core.utils import get_to_root_dir, get_filename_without_extension, generate_random_image
from src.ai.architectures import *  # Do not remove

base_config = {
    "architecture": "",
    "load_checkpoint_dir": None,
    "initialisation_type": InitializationType.Xavier,
    "initialisation_seed": 0,
    "device": 'cpu',
}


class ArchitectureTest(unittest.TestCase):

    def setUp(self) -> None:
        self.output_dir = f'{os.environ["PWD"]}/test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        base_config['output_path'] = self.output_dir

    def test_base_net(self):
        network = BaseNet(config=ArchitectureConfig().create(config_dict=base_config))
        msg = 'this test rocks.'
        network.save_to_checkpoint(extra_info={'msg': msg})
        network.load_from_checkpoint(os.path.join(self.output_dir, 'torch_checkpoints'))
        self.assertEqual(network.extra_checkpoint_info['msg'], msg)

    def test_tiny_128_rgb_1c_initialisation_store_load(self):
        # Test initialisation of different seeds
        base_config['architecture'] = 'tiny_128_rgb_1c'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        for p in network.parameters():
            check_network = p.data
            break
        base_config['initialisation_seed'] = 2
        second_network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        for p in second_network.parameters():
            check_second_network = p.data
            break
        self.assertNotEqual(torch.sum(check_second_network), torch.sum(check_network))
        base_config['initialisation_seed'] = 0
        third_network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        for p in third_network.parameters():
            check_third_network = p.data
            break
        self.assertEqual(torch.sum(check_third_network), torch.sum(check_network))
        # test storing and reloading
        second_network.save_to_checkpoint()
        network.load_from_checkpoint(os.path.join(self.output_dir, 'torch_checkpoints'))
        for p in network.parameters():
            check_network = p.data
            break
        self.assertEqual(torch.sum(check_second_network), torch.sum(check_network))
        self.assertNotEqual(torch.sum(check_third_network), torch.sum(check_network))

    def test_tiny_128_rgb_1c_input_failures(self):
        base_config['architecture'] = 'tiny_128_rgb_1c'
        network = eval(base_config['architecture']).Net(
            config=ArchitectureConfig().create(config_dict=base_config)
        )
        # try normal usage
        self.assertEqual(network.forward(generate_random_image(size=(3, 128, 128))).squeeze(0).shape,
                         network.output_size)

        # try channel last
        self.assertEqual(network.forward(generate_random_image(size=(128, 128, 3))).squeeze(0).shape,
                         network.output_size)
        # try batch normal usage
        self.assertEqual(network.forward(generate_random_image(size=(10, 3, 128, 128))).shape[1:],
                         network.output_size)
        # try batch channel last
        self.assertEqual(network.forward(generate_random_image(size=(10, 128, 128, 3))).shape[1:],
                         network.output_size)

    def tearDown(self) -> None:
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    get_to_root_dir()
    unittest.main()
