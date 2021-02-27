#!/usr/bin/python3.8
import os
import shutil

import rospy
from networkx.readwrite.tests.test_yaml import yaml
from sensor_msgs.msg import Image
from src.core.utils import get_filename_without_extension, get_data_dir
from src.core.data_types import ProcessState, Experience, TerminationType
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose
from src.sim.ros.src.utils import quaternion_from_euler
import numpy as np
from cv_bridge import CvBridge
import time
from scipy.spatial.transform import Rotation as R
from src.data.data_saver import DataSaver, DataSaverConfig
from src.core.config_loader import Config, Parser


# Class for data collection in a gazebo world.
class DataCollectionGazebo:

    def __init__(self):
        #self.output_dir = f'/media/thomas/Elements/experimental_data/loc_and_is_cone/{get_filename_without_extension(__file__)}'
        self.output_dir = f'{get_data_dir(os.environ["DATADIR"])}/norm_loc_and_is_cone/{get_filename_without_extension(__file__)}'
        print(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'random_seed': 123,
            'gazebo': 'true',
            'world_name': 'cone_world',
            'robot_name': 'drone_sim',
            'output_path': self.output_dir
        }
        # Initialise process wrapper ros
        self.ros_process = RosWrapper(launch_file='load_ros.launch',
                                      config=config,
                                      visible=False)
        # subscribe to the camera (expandable)
        subscribe_topics = [TopicConfig(topic_name=rospy.get_param(f'/robot/{sensor}_sensor/topic'),
                                        msg_type=rospy.get_param(f'/robot/{sensor}_sensor/type'))
                            for sensor in ['camera', 'position']]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=[]
        )
        self.bridge = CvBridge()

        # config of the data saver
        config_dict = {
            'output_path': self.output_dir,
            'separate_raw_data_runs': True,
            'store_hdf5': True
        }
        config_datasaver = DataSaverConfig().create(config_dict=config_dict)
        self._data_saver = DataSaver(config=config_datasaver)
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def set_model_state(self, name: str, position):
        model_state = ModelState()
        model_state.pose = Pose()
        model_state.model_name = name
        model_state.pose.position.x = position[0]
        model_state.pose.position.y = position[1]
        model_state.pose.position.z = position[2]
        roll = position[3]
        pitch = position[4]
        yaw = position[5]
        model_state.pose.orientation.x, model_state.pose.orientation.y, model_state.pose.orientation.z, \
        model_state.pose.orientation.w = quaternion_from_euler((roll, pitch, yaw))  # around x y z axis
        self._set_model_state(model_state)

        # send command to change drone pos
        # unpause client + take image
        # pause client

    def generate_image(self, total_data):
        model_name = rospy.get_param('/robot/model_name')
        print(model_name)
        self._set_model_state.wait_for_service()
        prev_seq_nb = -1
        print('waiting for first image')
        image0 = rospy.wait_for_message('/forward/camera/image', Image)  # test image
        prev_seq_nb = image0.header.seq
        print('first received')
        position_out = np.array([0.0, 0.0, 0.0])
        cone_in_image = 0
        for data_collect_amount in range(total_data):
            if np.random.rand() > 0.5:
                cone_in_image = 1
            else:
                cone_in_image = 0
            roll = -1 / 18 * np.pi + np.random.rand() / 9 * np.pi
            pitch = -1 / 18 * np.pi + np.random.rand() / 9 * np.pi + np.pi / 6  # 30 degrees tilt downwards looking on av
            yaw = -1 / 18 * np.pi + np.random.rand() / 9 * np.pi
            random_x = -9 * np.random.rand() - 1  # between 1 and 5 meters
            random_y = -random_x / 2 * np.random.rand() - 0.5
            random_z = -random_x / 2 * np.random.rand() + 0.5
            position = np.array([random_x, random_y, random_z, roll, pitch, yaw])
            # make changes in gazebo

            # cone or no cone
            if not cone_in_image:
                position = np.array([-1*random_x, random_y, random_z, roll, pitch, yaw])
            self._unpause_client(EmptyRequest())
            self.set_model_state(model_name, position)
            time.sleep(0.5)
            # check location: already adjusted?
            # unpause and record
            for sensor in ['camera']:  # collision < wrench, only published when turned upside down
                image = rospy.wait_for_message('/forward/camera/image', Image)
                position_gazebo = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                while image.header.seq == prev_seq_nb:
                    image = rospy.wait_for_message('/forward/camera/image', Image)
                    position_gazebo = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                # Read out the quadrotor positions
                index_drone = position_gazebo.name.index(model_name)
                pose_gazebo = position_gazebo.pose[index_drone].position
                quat = position_gazebo.pose[index_drone].orientation
                position_out[0] = pose_gazebo.x
                position_out[1] = pose_gazebo.y
                position_out[2] = pose_gazebo.z
                cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
                image_np = np.asarray(cv_im)
                # overcome saving double images, would generate a sequence of identical data
                print(image.header.seq)
                prev_seq_nb = image.header.seq
                # Generate an experience
                experience = Experience()
                experience.done = TerminationType.NotDone
                if data_collect_amount == total_data:
                    experience.done = TerminationType.Done

                ann_pos = self.generate_annotation_cone_from_quat(position_out, quat)
                experience.action = np.array([ann_pos[0],ann_pos[1],ann_pos[2],1])
                if not cone_in_image:
                    experience.action = np.array([0,0,0,0])
                experience.time_stamp = data_collect_amount
                experience.info = {"x": position[0], "y": position[1], "z": position[2], "yaw": position[5]}
                while image is None:
                    time.sleep(0.01)
                experience.observation = image_np
                self._data_saver.save(experience)
                image = None

                self._pause_client(EmptyRequest())

        self._data_saver.create_train_validation_hdf5_files()

    # Transforms the coordinates using the quaternions that describe the position of the drone
    def generate_annotation_cone_from_quat(self, position, quat):
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w])  # needs x,y,z,w format
        coordin_in = np.array([position[0], position[1], position[2]])
        r = R.from_quat(quat_np)
        matrix_r = r.as_matrix()
        coordin_out = -1 * coordin_in.dot(matrix_r)/10
        return coordin_out

    def generate_2d_annotations(self, annotations):
        return np.array([annotations[1] / annotations[0], annotations[2] / annotations[0]])

    def finish_collection(self):
        self.ros_process.terminate()


if __name__ == "__main__":
    arguments = Parser().parse_args()
    config_file = arguments.config
    if arguments.rm:
        with open(config_file, 'r') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        if not configuration['output_path'].startswith('/'):
            configuration['output_path'] = os.path.join(get_data_dir(os.environ['HOME']), configuration['output_path'])
        shutil.rmtree(configuration['output_path'], ignore_errors=True)

    data_col = DataCollectionGazebo()
    amount_of_images = 5
    data_col.generate_image(amount_of_images)
    data_col.finish_collection()
