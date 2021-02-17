#!/usr/bin/python3.8
import os
import rospy
from sensor_msgs.msg import Image

from src.core.utils import count_grep_name, get_data_dir
from src.core.utils import get_filename_without_extension
from src.core.data_types import ProcessState, Experience, TerminationType
from src.sim.ros.src.process_wrappers import RosWrapper
from src.sim.ros.test.common_utils import TopicConfig, TestPublisherSubscriber
from std_srvs.srv import Empty as Emptyservice, EmptyRequest
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist, Pose
from src.sim.ros.src.utils import quaternion_from_euler
import numpy as np
from cv_bridge import CvBridge
import time
from src.data.data_saver import DataSaver, DataSaverConfig

class DataCollectionGazebo:

    def __init__(self):
        self.output_dir = f'{get_data_dir(os.environ["CODEDIR"])}/{get_filename_without_extension(__file__)}'
        print(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        config = {
            'random_seed': 123,
            'gazebo': 'true',
            'world_name': 'cone_world',
            'robot_name': 'drone_sim',
            'output_path': self.output_dir
        }
        #initialise process wrapper ros
        self.ros_process = RosWrapper(launch_file='load_ros.launch',
                                      config=config,
                                      visible=False)
        # subscribe to the camera (expandable)
        subscribe_topics = [TopicConfig(topic_name=rospy.get_param(f'/robot/{sensor}_sensor/topic'),
                                        msg_type=rospy.get_param(f'/robot/{sensor}_sensor/type'))
                            for sensor in ['camera', 'position', 'depth']]
        self.ros_topic = TestPublisherSubscriber(
            subscribe_topics=subscribe_topics,
            publish_topics=[]
        )
        self.bridge = CvBridge()

        #config of the data saver
        config_dict = {
            'output_path': self.output_dir,
            'separate_raw_data_runs': False,
            'store_hdf5': True
        }
        config = DataSaverConfig().create(config_dict=config_dict)
        self._data_saver = DataSaver(config=config)
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)


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

        # send command to change drone pos
        # unpause client + take image
        # pause client
    def generate_image(self):
        model_name = rospy.get_param('/robot/model_name')
        total_data = 100  # generate 10 fragments/pictures
        self._set_model_state.wait_for_service()
        position = np.array([2, 0, 1, np.pi, 0, 0])
        time.sleep(5) #setup ros
        prev_seq_nb = 0
        image0 = rospy.wait_for_message('/forward/camera/image', Image) #test image
        for data_collect_amount in range(total_data):
            yaw = 0
            random_distance = 5 * np.random.rand() + .5
            position = np.array([random_distance, 0, 1, yaw + np.pi, 0, 0])
            # make changes in gazebo
            self.set_model_state(model_name, position)
            # check location: already adjusted?
            # unpause and record
            self._unpause_client(EmptyRequest())
            for sensor in ['camera']:  # collision < wrench, only published when turned upside down
                image = rospy.wait_for_message('/forward/camera/image', Image)
                while image.header.seq == prev_seq_nb:
                    image = rospy.wait_for_message('/forward/camera/image', Image)
                    time.sleep(0.01)

                cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
                image_np = np.asarray(cv_im)
                print(image.header.seq)
                prev_seq_nb = image.header.seq
                experience = Experience()
                experience.done = TerminationType.NotDone
                if data_collect_amount == total_data:
                    experience.done = TerminationType.Done
                experience.action = self.generate_annotation_cone(position)
                experience.time_stamp = data_collect_amount
                experience.info = {"x": position[0], "y": position[1], "z": position[2]}
                while image is None:
                    time.sleep(0.1)
                experience.observation = image_np
                self._data_saver.save(experience)
                image = None
            self._pause_client(EmptyRequest())

        self._data_saver.create_train_validation_hdf5_files()
        # pause again and loop
        self.finish_collection()

    def generate_annotation_cone(self, position):
        distance = np.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
        alfa = position[3]-np.pi
        beta = 0
        x = np.cos(alfa) * distance
        y = np.sin(beta) * distance
        z = np.sin(alfa) * distance
        return np.array([x, y, z])

    def finish_collection(self):
        self.ros_process.terminate()


if __name__ == "__main__":
    data_col = DataCollectionGazebo()
    data_col.generate_image()