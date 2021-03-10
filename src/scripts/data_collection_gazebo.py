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
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel, DeleteModel
from src.sim.ros.src.utils import quaternion_from_euler_ZYX
import numpy as np
from cv_bridge import CvBridge
import time
from scipy.spatial.transform import Rotation as R
from src.data.data_saver import DataSaver, DataSaverConfig
from src.core.config_loader import Config, Parser


# Class for data collection in a gazebo world.
class DataCollectionGazebo:

    def __init__(self):
        # self.output_dir = f'/media/thomas/Elements/experimental_data/dept_est/{get_filename_without_extension(__file__)}'
        self.output_dir= f'/esat/opal/r0667559/data'
        # self.output_dir = f'{get_data_dir(os.environ["DATADIR"])}/cone_data/{get_filename_without_extension(__file__)}'
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
        self.prev_zone = 1
        # config of the data saver
        config_dict = {
            'output_path': self.output_dir,
            'separate_raw_data_runs': True,
            'store_hdf5': True
        }
        # Create data saver
        config_datasaver = DataSaverConfig().create(config_dict=config_dict)
        self._data_saver = DataSaver(config=config_datasaver)
        # Create proxy for needed services.
        print('create proxys')
        rospy.wait_for_service('/gazebo/unpause_physics')
        self._unpause_client = rospy.ServiceProxy('/gazebo/unpause_physics', Emptyservice)
        rospy.wait_for_service('/gazebo/pause_physics')
        self._pause_client = rospy.ServiceProxy('/gazebo/pause_physics', Emptyservice)
        rospy.wait_for_service('/gazebo/set_model_state')
        self._set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        self._spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        rospy.wait_for_service('gazebo/delete_model')
        self._delete_model = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        print('proxys initialised')

    # Collection of data in simulation environment gazebo.
    # Data is saved as experiments and written to the output folder.
    # Input: amount of images to produce.
    def generate_image(self, total_data,create_hdf5=False):
        model_name = rospy.get_param('/robot/model_name')
        cone_name = 'Cone'
        self._set_model_state.wait_for_service()
        print('waiting for first image')
        image0 = rospy.wait_for_message('/forward/camera/image', Image)  # test image
        prev_seq_nb = image0.header.seq
        # Main loop to create images.
        for data_collect_amount in range(total_data):
            if np.random.rand() > 0.5:
                self.delete_model('Boxed_flyzone')
                self.spawn_drone_room(np.array([-2.5, 0, 0]))
            # make changes in gazebo
            position = self.get_random_circle_position()
            self._unpause_client(EmptyRequest())
            self.set_model_state(model_name, position)
            time.sleep(0.5)
            # Collection of annotated data.
            for sensor in ['camera']:  # collision < wrench, only published when turned upside down
                image = rospy.wait_for_message('/forward/camera/image', Image)
                position_gazebo = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                while image.header.seq == prev_seq_nb:
                    image = rospy.wait_for_message('/forward/camera/image', Image)
                    position_gazebo = rospy.wait_for_message('/gazebo/model_states', ModelStates)
                # Read out the quadrotor positions
                index_drone = position_gazebo.name.index(model_name)
                index_cone = not not position_gazebo.name.index(cone_name)
                pose_gazebo_drone = position_gazebo.pose[index_drone].position
                quat = position_gazebo.pose[index_drone].orientation
                pose_gazebo_cone = position_gazebo.pose[index_cone].position
                position_out = np.array([pose_gazebo_drone.x, pose_gazebo_drone.y, pose_gazebo_drone.z])
                position_cone = np.array([pose_gazebo_cone.x, pose_gazebo_cone.y, pose_gazebo_cone.z])
                cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough') #process image
                image_np = np.asarray(cv_im)
                # overcome saving double images, would generate a sequence of identical data
                print(image.header.seq)
                prev_seq_nb = image.header.seq
                # Generate an experience
                experience = Experience()
                experience.done = TerminationType.NotDone
                if data_collect_amount == total_data:
                    experience.done = TerminationType.Done
                experience.action = self.generate_annotation_cone_from_quat(position_out, position_cone, quat)
                experience.time_stamp = data_collect_amount
                experience.info = {"x": position[0]}
                while image is None:
                    time.sleep(0.01)
                experience.observation = image_np
                self._data_saver.save(experience)
                image = None

                self._pause_client(EmptyRequest())
        # At the end, create hdf5_files
        if create_hdf5:
            self._data_saver.create_train_validation_hdf5_files()

    # Transforms the coordinates using the quaternions that describe the position of the drone
    def generate_annotation_cone_from_quat(self, position_drone, position_cone, quat):
        quat_np = np.array([quat.x, quat.y, quat.z, quat.w])  # needs x,y,z,w format
        coordin_in = np.array([position_drone[0]-position_cone[0], position_drone[1]-position_cone[1], position_drone[2]-position_cone[2]])
        r = R.from_quat(quat_np)
        matrix_r = r.as_matrix()
        coordin_out = -1 * coordin_in.dot(matrix_r)
        return coordin_out

    # Generates the annotations in an image. Useful for debugging.
    def generate_2d_annotations(self, annotations):
        return np.array([annotations[1] / annotations[0], annotations[2] / annotations[0]])

    def finish_collection(self):
        self.ros_process.terminate()

    # Samples random camera position facing to the (0,0,0) coordinate.
    def get_random_position(self):
        roll = -np.pi/6 -1 / 18 * np.pi + np.random.rand() / 9 * np.pi
        pitch = 0 -1 / 18 * np.pi + np.random.rand() / 9 * np.pi   # 15 degrees tilt downwards looking on av
        yaw = np.pi/2 + np.random.rand() / 9 * np.pi
        random_y = -10 * np.random.rand() - 2  # between 2 and 5 meters
        random_x = -random_y / 2 * (np.random.rand() - 0.5)
        random_z = -random_y / 2 * np.random.rand() + 1
        position = np.array([random_x, random_y, random_z, roll, pitch, yaw])
        return position

    # Samples random camera position around a given centre which is a cone.
    # The camera is rotated to have the cone in sight.
    def get_random_circle_position(self, centre = np.array([0,0])):
        distance = np.random.rand() * 5 + 1.5
        yaw = np.random.rand() * 2 * np.pi
        random_x = -np.cos(yaw)*distance + centre[0] + (np.random.rand()-0.5) * distance / 4
        random_y = -np.sin(yaw)*distance + centre[1] + (np.random.rand()-0.5) * distance / 4
        yaw_randomized = yaw + (np.random.rand()-0.5) * np.pi / 6
        random_z = np.random.rand() * 2 + 1
        roll = (np.random.rand()-.5) * np.pi/6
        pitch = np.pi/8 + (np.random.rand()-0.5) * np.pi/6
        position = np.array([random_x, random_y, random_z, roll, pitch, yaw_randomized])
        return position

    def spawn_drone_room(self, location):
        postition = Pose()
        postition.position.x = location[0]
        postition.position.y = location[1]
        postition.position.z = location[2]
        # read sdf file
        path_model_drone_zone = f'src/sim/ros/gazebo/models/flyzone_wall/flyzone_wall.sdf'
        path_model_street = f'src/sim/ros/gazebo/models/GE_flyzone/flyzone_GE.sdf'
        if not self.prev_zone:
            path_model = path_model_drone_zone
            self.prev_zone = 1
        else:
            self.prev_zone = 0
            path_model = path_model_street
        file_open = open(path_model,'r')
        sdff = file_open.read()
        self._spawn_model_prox("Boxed_flyzone", sdff, "flyzone_GE", postition, "world") #TODO make dynamic naming, hold in dict?

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
            model_state.pose.orientation.w = quaternion_from_euler_ZYX((yaw, pitch, roll))  # around x y z axis
        self._set_model_state(model_state)

    def delete_model(self, model_name):
        self._delete_model(model_name)


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
    amount_of_images = 50
    data_col.generate_image(amount_of_images,False)
    data_col.finish_collection()