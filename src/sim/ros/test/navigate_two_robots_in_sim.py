import os
import shutil
import unittest

from src.core.utils import get_filename_without_extension
from src.core.data_types import TerminationType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState


class TestRobots(unittest.TestCase):

    def start(self,
              robot_name: str,
              fsm_config: str = 'single_run') -> None:
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'
        os.makedirs(self.output_dir, exist_ok=True)
        config_dict = {
            'output_path': self.output_dir,
            'factory_key': "ROS",
            'max_number_of_steps': -1,
            'ros_config': {
                'ros_launch_config': {
                    'control_mapping_config': 'keyboard_and_joystick',
                    'fsm_config': fsm_config,
                    'gazebo': 'sim' in robot_name,
                    'random_seed': 123,
                    'robot_name': robot_name,
                    'world_name': 'debug_turtle' if 'sim' in robot_name else 'empty',
                    'robot_display': True,
                    'x_pos': 0.0,
                    'y_pos': 0.0,
                    'yaw_or': 0.0,
                    'z_pos': 0.0,
                },
                'actor_configs': [
                    {
                        'file': f'src/sim/ros/config/actor/keyboard_drone_sim.yml',
                        'name': 'keyboard'
                    },
                    {
                        'file': f'src/sim/ros/config/actor/joystick_drone_sim.yml',
                        'name': 'joystick'
                    }
                ],
                'visible_xterm': True,
            },
        }
        config = EnvironmentConfig().create(config_dict=config_dict)
        self._environment = RosEnvironment(config=config)

    def test_double_drone_sim(self):
        self.start(robot_name='double_drone_sim', fsm_config='single_run')
        self._environment.reset()
        while True:
            self._environment.step()

    def tearDown(self) -> None:
        if hasattr(self, '_environment'):
            self._environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
