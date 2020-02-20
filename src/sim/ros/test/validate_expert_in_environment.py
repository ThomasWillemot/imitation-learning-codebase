import os
import shutil
import unittest

import rospy
import yaml

from src.core.utils import get_filename_without_extension
from src.sim.common.data_types import TerminalType
from src.sim.common.environment import EnvironmentConfig
from src.sim.ros.src.ros_environment import RosEnvironment
from src.sim.ros.catkin_ws.src.imitation_learning_ros_package.rosnodes.fsm import FsmState


class TestRosExpert(unittest.TestCase):

    def setUp(self) -> None:
        #  Define the environment you want to test:
        world_name = 'cube_world'
        self.output_dir = f'test_dir/{get_filename_without_extension(__file__)}'

        config_file = 'test_expert_in_environment'
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f'src/sim/ros/test/config/{config_file}.yml', 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        config_dict['output_path'] = self.output_dir
        config_dict['ros_config']['ros_launch_config']['world_name'] = world_name
        self.environment_config = EnvironmentConfig().create(
            config_dict=config_dict
        )

    def test_expert(self):
        self.environment = RosEnvironment(
            config=self.environment_config
        )
        state = self.environment.reset()

        # wait delay evaluation time
        while state.terminal == TerminalType.Unknown:
            state = self.environment.step()
        print(f'finished startup')
        waypoints = rospy.get_param('/world/waypoints')

        for waypoint_index, waypoint in enumerate(waypoints[:-1]):
            print(f'started with waypoint: {waypoint}')
            while state.sensor_data['current_waypoint'].tolist() == waypoint:
                state = self.environment.step()

        print(f'ending with waypoint {waypoints[-1]}')
        while not self.environment.fsm_state == FsmState.Terminated:
            state = self.environment.step()
        print(f'terminal type: {state.terminal.name}')

    def tearDown(self) -> None:
        self.environment.remove()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
