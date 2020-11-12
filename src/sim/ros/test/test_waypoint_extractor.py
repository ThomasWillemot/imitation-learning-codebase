import os
import shutil
import time
import unittest

import numpy as np
import rospy
from imitation_learning_ros_package.srv import SendRelCor
from src.sim.ros.src.process_wrappers import RosWrapper

#file to test functions of waypoint_extractor. Both testing the service and the coordinate extraction.

class TestWaypointExtractor(unittest.TestCase):
    self.output_dir = f'{get_data_dir(os.environ["HOME"])}/test_dir/{get_filename_without_extension(__file__)}'
    print(self.output_dir)
    os.makedirs(self.output_dir, exist_ok=True)

    config = {
        'output_path': self.output_dir,
        'world_name': 'test_robot_mapper',
        'robot_name': 'turtlebot_sim',
        'gazebo': False,
        'fsm': False,
        'control_mapping': False,
        'ros_expert': False,
        'waypoint_indicator': False,
        'robot_mapping': False
    }

    # spinoff roslaunch
    self._ros_process = RosWrapper(launch_file='load_ros.launch',
                                   config=config,
                                   visible=False)

    def start_test(self):
        rospy.wait_for_service('rel_cor')
        try:
            get_rel_cor = rospy.ServiceProxy('rel_cor', SendRelCor)
            resp1 = get_rel_cor()
            return [resp1.x, resp1.y, resp1.z]
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

if __name__ == '__main__':
    unittest.main()
    print("Requesting COOR")
    print(start_test())
