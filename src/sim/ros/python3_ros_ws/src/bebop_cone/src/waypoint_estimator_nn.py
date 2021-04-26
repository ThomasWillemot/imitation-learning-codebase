#!/usr/bin/python3.8
"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import os
from glob import glob

import numpy as np
import torch
from cv_bridge import CvBridge
import rospy
import cv2

from sensor_msgs.msg import *
from tf2_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs import *
from std_srvs.srv import Trigger
from bebop_cone.srv import SendRelCor, SendRelCorResponse
from bebop_cone.msg import ConeImgLoc
from src.sim.ros.python3_ros_ws.src.bebop_cone.helper_files import cnn_architecture
from src.sim.ros.python3_ros_ws.src.bebop_cone.helper_files.ArchitectureConfig import ArchitectureConfig


class WaypointEstimatorNN:

    def __init__(self):

        rospy.init_node('waypoint_extractor_server')
        self.bridge = CvBridge()
        dim = (848, 800)
        self.threshold = 140  # TODO change if needed using rviz (check images)
        #TODO FIND K AND D FOR CAM
        k = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        d = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        self.x = 0
        self.y = 0
        self.z = 0
        self.kernel = np.ones((3, 3), np.uint8)
        self.rate = rospy.Rate(1000)
        self.image1_buffer = []
        self.image2_buffer = []
        self.image_stamp = rospy.Time(0)
        self.running_average = np.array([0, 0, 0, 0, 0, 0])
        self.original_model_device = 'default'
        #TODO enable if on drone self._init_fsm_handshake_srv()
        self.pub = rospy.Publisher('cone_coordin', ConeImgLoc, queue_size=10)
        self.thresh_pub = rospy.Publisher('threshold_im', Image, queue_size=10)
        architecture_config = ArchitectureConfig()
        self.trainer = None
        self.environment = None
        self.epoch = 0
        self.net = eval('cnn_architecture').Net(config=architecture_config) \
            if architecture_config is not None else None
        self.load_checkpoint('/media/thomas/Elements/training_nn/res_200/6100_lr_0002')
        self.put_model_on_device('cuda')
    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, current_image, threshold):
        binary_image = cv2.threshold(current_image, threshold, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        return binary_image[1]

    def put_model_on_device(self, device: str = None):
        original_model_device = self.net.get_device()
        torch.device(device)

    def put_model_back_to_original_device(self):
        self.net.set_device(self.original_model_device)

    def downsample_image(self,image, factor=1):
        img = np.array(image, dtype='float32')
        img = torch.from_numpy(img.reshape(1, 1, img.shape[0], img.shape[1]))  # Convert grayscale image to tensor
        maxPool = torch.nn.AvgPool2d(factor)  # 4*4 window, maximum pooling with a step size of 4
        img_tensor = maxPool(img)
        #img = torch.squeeze(img)  # Remove the dimension of 1
        #img = img.numpy().astype('uint8')  # Conversion format, ready to output
        return img_tensor
    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint(self, image):
        print("Estimate wp")
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv
        rect_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        # Cone segmentation

        bin_im = self.get_cone_binary(rect_image, threshold=self.threshold)
        post_proc_im = self.post_process_image(bin_im)
        down_tens_image = self.downsample_image(post_proc_im, factor=4)
        # Positioning in 2D of cone parts
        cone_coordinates = self.eval_neural_net(down_tens_image)
        # Use scaling factor ( neural net is trained in other way than images are.
        self.running_average = self.running_average*0.65+cone_coordinates*0.35
        cone_coordinates = self.running_average
        x_position = int(-cone_coordinates[1] / cone_coordinates[0] * 286 + 418)
        y_position = int(-cone_coordinates[2] / cone_coordinates[0] * 286 + 405)
        x_2_position = int(-cone_coordinates[4] / cone_coordinates[3] * 286 + 418)
        y_2_position = int(-cone_coordinates[5] / cone_coordinates[3] * 286 + 405)
        self.image_publisher(x_position, y_position, 50/cone_coordinates[0])
        self.threshold_image_publish(bin_im, x_position, y_position, 50/cone_coordinates[0], x_2=x_2_position,y_2=y_2_position,size_2=50/cone_coordinates[3])
        return cone_coordinates

    def eval_neural_net(self, image):
        predictions = self.net.forward(image, train=False)
        np_pred = predictions.detach().numpy()
        return np_pred[0]

    def load_checkpoint(self, checkpoint_dir: str):
        if not checkpoint_dir.endswith('torch_checkpoints'):
            checkpoint_dir += '/torch_checkpoints'
        if len(glob(f'{checkpoint_dir}/*.ckpt')) == 0 and len(glob(f'{checkpoint_dir}/torch_checkpoints/*.ckpt')) == 0:
            raise FileNotFoundError
        # Get checkpoint in following order
        if os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_best.ckpt')
        elif os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')):
            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint_latest.ckpt')
        else:
            checkpoints = {int(f.split('.')[0].split('_')[-1]): os.path.join(checkpoint_dir, f)
                           for f in os.listdir(checkpoint_dir)}
            checkpoint_file = checkpoints[max(checkpoints.keys())]
        # Load params for each experiment element
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else 0
        for element, key in zip([self.net, self.trainer, self.environment],
                                ['net_ckpt', 'trainer_ckpt', 'environment_ckpt']):
            if element is not None and key in checkpoint.keys():
                element.load_checkpoint(checkpoint[key])
        print('checkpoint loaded')

    def post_process_image(self, image, binary=False):
        height = 800
        width = 848
        row_sum = np.sum(image, axis=1)  # should be 800 high

        airrow = 0
        for row_idx in range(799):
            if row_sum[row_idx] > 400 * 255:
                airrow = row_idx
        image[0:airrow, :] = 0
        i = airrow
        prev_empty = False
        while i < 799:
            curr_empty = row_sum[i] > 255
            if curr_empty:
                image[i, :] = np.zeros(848)
            elif prev_empty:
                break
            else:
                prev_empty = True
            i += 1
        image_np_gray = np.asarray(image)
        if np.amax(image_np_gray) == 255:
            image_np_gray = image_np_gray/255
        filtered_np_gray = cv2.morphologyEx(image_np_gray, cv2.MORPH_OPEN, self.kernel)
        return filtered_np_gray


    # Handles the service requests.
    def handle_cor_req(self, req):
        print("STAMP")
        print(self.image_stamp)
        # TEST DUMMY - REMOVE THIS
        # coor = [0, 3, 4]
        return SendRelCorResponse(self.running_average[0],self.running_average[1],self.running_average[2],self.running_average[3],self.running_average[4],self.running_average[5], self.image_stamp)

    def _init_fsm_handshake_srv(self):
        """Setup handshake service for FSM.
        """
        self.fsm_handshake_srv = rospy.Service(
            "/waypoint_extractor_server/fsm_handshake", Trigger, self.fsm_handshake)

    def fsm_handshake(self, _):
        '''Handles handshake with FSM. Return that initialization was successful and
	    waypoint exctractor is running.
        '''
        return {"success": True, "message": ""}

    def rel_cor_server(self):
        '''Service for delivery of current relative coordinates
        '''
        s = rospy.Service('/waypoint_extractor_server/rel_cor', SendRelCor, self.handle_cor_req)
        rospy.loginfo("WPE  - Waypoint extractor running. Waiting for request")

    def image_subscriber(self):
        '''Subscribes to topics and and runs callbacks
        '''
        # These always come with identical timestamps. Callbacks at slightly offset times.
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.fisheye1_callback)

    def fisheye1_callback(self, image):
        '''Buffer images coming from /camera/fisheye1/image_raw. Buffer is cleared in run().
        Args:
            image: std_msgs/Image
        '''
        self.image1_buffer.append(image)

    '''Augments the grayscale or binary image
    Args:
        image: bin numpy image
        max_start: image coordinate u for the circle
        cone_row: image coordinate v for the circle
        max_width: width of the circle
    '''
    def threshold_image_publish(self, image, x, y, size, x_2, y_2, size_2):
        resolution = (800, 848)
        frame = np.array(image)
        frame = cv2.circle(frame, (x, y), int(max(size, 2)/2), 255, 2)
        frame = cv2.circle(frame, (x_2, y_2), int(max(size_2, 2) / 2), 255, 2)
        image = Image()
        image.data = frame.astype(np.uint8).flatten().tolist()
        image.height = resolution[0]
        image.width = resolution[1]
        image.encoding = 'mono8'
        image.step = resolution[1]
        self.thresh_pub.publish(image)

    def image_publisher(self, x_coor, y_coor, width):
        cone_coor1 = ConeImgLoc()
        cone_coor1.x_pos = np.int32(x_coor)
        cone_coor1.y_pos = np.int32(y_coor)
        cone_coor1.cone_width = np.int16(int(max(width, 2)))
        self.pub.publish(cone_coor1)

    def run(self):
        '''Starts all needed functionalities + Main loop
        '''
        self.image_subscriber()
        self.rel_cor_server()

        while not rospy.is_shutdown():
            if self.image1_buffer:
                image1 = self.image1_buffer.pop()
                print("pop")
                print(image1.header.stamp.to_sec())

                self.image1_buffer.clear()
                self.image2_buffer.clear()

                relat_coor = self.extract_waypoint(image1)

                if 5 > relat_coor[0] > 0:  # only update if in range of 5 meter
                    self.update_median(relat_coor)
                print('Coordinates')
                print(self.x, self.y, self.z)
                self.image_stamp = image1.header.stamp

            self.rate.sleep()

        # rospy.spin()


if __name__ == "__main__":
    waypoint_estimator_nn = WaypointEstimatorNN()
    waypoint_estimator_nn.run()
