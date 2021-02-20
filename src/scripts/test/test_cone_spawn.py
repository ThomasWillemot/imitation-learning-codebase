#!/usr/bin/python3.8
import os
import shutil
import unittest
import time

import rospy
from sensor_msgs.msg import Image

from src.core.utils import count_grep_name, get_data_dir
from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState, Experience
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose
from src.sim.ros.src.utils import quaternion_from_euler
import numpy as np
from cv_bridge import CvBridge
from src.data.data_saver import DataSaver, DataSaverConfig


class TestConeSpawn(unittest.TestCase):
    """
    Basic test that validates position, depth, camera sensors are updated
    """

    def setUp(self) -> None:
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/test_dir/{get_filename_without_extension(__file__)}'
        print(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'random_seed': 123,
            'gazebo': 'true',
            'world_name': 'cone_world',
            'robot_name': 'drone_sim',
            'output_path': self.output_dir
        }
        self.ros_process = RosWrapper(launch_file='load_ros.launch',
                                      config=config,
                                      visible=False)
        subscribe_topics = [TopicConfig(topic_name=rospy.get_param(f'/robot/{sensor}_sensor/topic'),
                                        msg_type=rospy.get_param(f'/robot/{sensor}_sensor/type'))
                            for sensor in ['camera', 'position', 'depth']]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=[]
        )
        self.bridge = CvBridge()

        config_dict = {
            'output_path': self.output_dir,
            'separate_raw_data_runs': False,
            'store_hdf5': True,
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self._data_saver = DataSaver(config=config)
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)




    # send command to change drone pos
    # unpause client + take image
    # pause client
    def test_generate_image(self):
        model_name = rospy.get_param('/robot/model_name')
        total_data = 10  # generate 10 fragments/pictures
        self._set_model_state.wait_for_service()
        position = np.array([2, 0, 1, np.pi, 0, 0])
        time.sleep(10)
        for data_collect_amount in range(total_data):
            time.sleep(0.1)
            self.set_model_state(model_name, position)
            random_distance = 5*np.random.rand()+.5
            print(random_distance)
            yaw = 0
            
            position = np.array([random_distance, 0, 1, yaw +np.pi, 0, 0])
            # check location: already adjusted?
            # unpause and record
            self._unpause_client(EmptyRequest())
            print(self._data_saver.get_saving_directory())
            for sensor in ['camera']:  # collision < wrench, only published when turned upside down
                print(sensor)
                print(rospy.get_param(f'/robot/{sensor}_sensor/topic') in self.ros_topic.topic_values.keys())
                self.assertTrue(rospy.get_param(f'/robot/{sensor}_sensor/topic') in self.ros_topic.topic_values.keys())
                image = rospy.wait_for_message('/forward/camera/image', Image)
                cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
                image_np = np.asarray(cv_im)
                experience = Experience()
                experience.action = self.generate_annotation_cone(position)
                experience.observation = image_np
                experience.time_stamp = data_collect_amount
                experience.info = {"x": position[0], "y": position[1], "z": position[2]}
                self._data_saver.save(experience)
            self._pause_client(EmptyRequest())

        self._data_saver.create_train_validation_hdf5_files()
            # pause again and loop

    def set_model_state(self,name: str, position):
        model_state = ModelState()
        model_state.pose = Pose()
        model_state.model_name = name
        model_state.pose.position.x = position[0]
        model_state.pose.position.y = position[1]
        model_state.pose.position.z = position[2]
        yaw = position[3]
        model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, \
        model_state.pose.orientation.w = quaternion_from_euler((0, 0, yaw))
        self._set_model_state(model_state)

    def test_generate_annotation_cone(self):
        position = np.array([-1, 0, 1, 0, 0, 0])
        annotations = self.generate_annotation_cone(position)
        self.assertEqual(annotations[0], 1)
        self.assertEqual(annotations[1], 0)
        self.assertEqual(annotations[2], -1)

    # Generate the cone 3d location from the angles and distance.
    # Rotations are performed around XYZ, in that order
    def generate_annotation_cone(self, position):
        yaw = position[5]
        roll = position[3]
        pitch = position[4]

        rotation_roll = np.array([[1, 0, 0], [0, np.cos(roll), -1 * np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

        rotation_pitch = np.array(
            [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-1 * np.sin(pitch), 0, np.cos(pitch)]])

        rotation_yaw = np.array([[np.cos(yaw), -1 * np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = np.matmul(np.matmul(rotation_roll, rotation_pitch), rotation_yaw)
        coordin_in = np.array([-1 * position[0], -1 * position[1], -1 * position[2]])
        coordin_out = coordin_in.dot(rotation_matrix)
        return coordin_out

    def tearDown(self) -> None:
        self.ros_process.terminate()
        shutil.rmtree(self.output_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
