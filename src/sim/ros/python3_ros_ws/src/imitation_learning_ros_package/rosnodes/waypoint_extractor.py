#!/usr/bin/python3.8

"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import os
import sys
import time
import numpy as np
import rospy
from sensor_msgs.msg import *
#from nav_msgs.msg import Odometry
sys.path.append('src/sim/ros/test')
from std_msgs.msg import *
from imitation_learning_ros_package.srv import SendRelCor, SendRelCorResponse
from common_utils import get_fake_image
import cv2
#from src.core.logger import get_logger, cprint, MessageType
#from src.core.utils import camelcase_to_snake_format, get_filename_without_extension
#from src.sim.ros.src.utils import get_output_path


class WaypointExtractor:

    def __init__(self):
        start_time = time.time()
        max_duration = 60
        # Wait for parameters to load
        #while not rospy.has_param('/robot/odometry_topic') and time.time() < start_time + max_duration:
        #    time.sleep(0.1)
        #self._output_path = get_output_path()
        #self._logger = get_logger(get_filename_without_extension(__file__), self._output_path)
        self.current_image = []
        self.x_cor = 0
        self.y_cor = 0
        self.z_cor = 0
        rospy.init_node('waypoint_extractor_server')
        #s = rospy.Service('waypoin_extractor', ExtractWaypoint, hand)
        #rospy.spin()

    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, cone_color):
        image_shape = self.current_image.shape
        print(image_shape)
        binary_image = np.zeros((image_shape[0], image_shape[1]))
        threshold_red_gr = cone_color[2] / cone_color[1] * .9
        threshold_green_bl = cone_color[1] / cone_color[0] * 0.9
        for i in range(image_shape[0]):
            for k in range(image_shape[1]):
                if threshold_red_gr < self.current_image[i, k, 2] / self.current_image[i, k, 1]:
                    if threshold_green_bl < self.current_image[i, k, 1] / self.current_image[i, k, 0]:
                        binary_image[i, k] = 1
        return binary_image

    def get_cone_2d_location(self, bin_im):
        im_size = bin_im.shape
        x_starts = []
        y_starts = []
        widths = []
        max_width_row = 0
        max_width_start = -1
        current_width_start = -1
        current_width = 0
        prev = 0
        for row in range(im_size[0]):
            for hor_pix in range(im_size[1]):
                if current_width == 0:  # This means a new sequence starts
                    if bin_im[row, hor_pix] == 1:
                        current_width_start = hor_pix
                        current_width = 1
                else:
                    if bin_im[row, hor_pix] == 1:
                        if prev == 1:
                            current_width += 1
                        prev = 1
                    else:
                        if prev == 0:
                            if bin_im[row, hor_pix - 2] == 0 and bin_im[row, hor_pix - 3] == 0:
                                if max_width_row < current_width:
                                    max_width_row = current_width
                                    max_width_start = current_width_start
                                current_width_start = -1
                                current_width = 0
                        prev = 0
            if max_width_row > 20:
                x_starts.append(max_width_start - im_size[1] / 2 + max_width_row / 2)
                y_starts.append(-row + im_size[0] / 2)
                widths.append(max_width_row)
            current_width_start = -1
            current_width = 0
            max_width_start = -1
            max_width_row = 0
            if not widths:
                widths.append(0)
                x_starts.append(-1)
                y_starts.append(-1)

        return [x_starts, y_starts, widths]

    def get_cone_3d_location(self, cone_width_px, cone_width_m, prev_loc, conetop_coor, tune_factor):
        # position relative to the camera in meters.
        if conetop_coor[0] == -1 and conetop_coor[1] == -1:
            self.x_cor = prev_loc[0]
            self.y_cor = prev_loc[1]
            self.z_cor = prev_loc[2]
        else:
            self.z_cor = cone_width_m * tune_factor / cone_width_px
            self.x_cor = conetop_coor[0] * self.z_cor / tune_factor
            self.y_cor = conetop_coor[1] * self.z_cor / tune_factor

    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint(self, image):
        image_height = image.height
        image_width = image.width
        deserialized_bytes = np.frombuffer(image.data, dtype=np.int8)
        self.current_image = np.reshape(deserialized_bytes, (image_height, image_width, -1))
        print(self.current_image.shape)
        bin_im = self.get_cone_binary(cone_color=np.array([50, 50, 200]))
        loc_2d = self.get_cone_2d_location(bin_im)
        max_width = max(loc_2d[2]) #Max width detected, assumption biggest object is the cone.
        max_index = loc_2d[2].index(max_width) # Index of middle of max width
        self.get_cone_3d_location(max_width, 0.14, [0, 1, 5], [loc_2d[0][max_index], loc_2d[1][max_index]], 1585)
        # tunefactor calculated by distance[m]*pixels of ob/seize obj[m]

    def handle_cor_req(self, req):
        print("This function should answer a called service request with the coordinates.")
        print([self.x_cor, self.y_cor, self.z_cor])
        return SendRelCorResponse(self.x_cor, self.y_cor, self.z_cor)

    def rel_cor_server(self):
        s = rospy.Service('rel_cor', SendRelCor, self.handle_cor_req)
        print("Waiting for request")


    def image_subscriber(self):
        rospy.Subscriber("image_topic", Image, self.extract_waypoint)

    def run(self):
        print("Should start the waypoint extrac function to extract images out of the 3d coordinates")
        self.image_subscriber()
        self.rel_cor_server()
        print("Should start the service to deliver 3d coordinates")
        rospy.spin()
# rospy.wait_for_service('/enable_motors')
# enable_motors_service = rospy.ServiceProxy('/enable_motors', EnableMotors)
# enable_motors_service.call(True)
if __name__ == "__main__":
    waypoint_extractor = WaypointExtractor()
    waypoint_extractor.run()

