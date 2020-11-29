#!/usr/bin/python3.8
"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import numpy as np
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import *
from tf2_msgs.msg import *
sys.path.append('src/sim/ros/test')
from std_msgs.msg import *
from imitation_learning_ros_package.srv import SendRelCor, SendRelCorResponse
import cv2


class WaypointExtractor:

    def __init__(self):
        self.bridge = CvBridge()
        self.DIM = (800, 848)
        self.K = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        self.D = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM,
                                                                   cv2.CV_16SC2)
        self.counter = 0
        self.x_cor = 0
        self.y_cor = 0
        self.z_cor = 0
        self.image_rec = 0
        rospy.init_node('waypoint_extractor_server')

    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, current_image, treshold):
        image_shape = current_image.shape
        binary_image = np.zeros((image_shape[0], image_shape[1]))
        for i in range(image_shape[0]):  # Each row
            for k in range(image_shape[1]):  # Each column
                if current_image[i, k] > treshold:
                    binary_image[i, k] = 255
        return binary_image

    # Extract the 2d location in the image after segmentation.
    # TODO: loops of get_cone_binary and this function can be written in one.

    def get_cone_2d_location(self, bin_im):
        cone_found = False
        im_size = bin_im.shape
        max_width_row = 0
        max_width_x = -1
        max_width_y = -1
        current_width_start = -1
        current_width = 0
        prev_pix = 0
        row = im_size[0] - 1
        prev_row = 0
        while not cone_found and row >= 0:
            for column in range(im_size[1]):
                if bin_im[row, column] > 0:
                    if current_width_start == -1:
                        current_width_start = column
                    elif prev_pix == 1:
                        current_width += 1
                    elif current_width == 0:
                        current_width_start = column
                    prev_pix = 1
                else:
                    prev_pix = 0
            if current_width > max_width_row and current_width_start > 0:
                max_width_row = current_width
                max_width_x = current_width_start
                max_width_y = row
            if prev_row == 1 and current_width_start == -1:
                cone_found = True
            if current_width_start > -1:
                prev_row = 1
            current_width = 0
            current_width_start = -1
            row -= 1
        return [max_width_x-424, -max_width_y+400, max_width_row]

    def get_cone_3d_location(self, cone_width_px, cone_width_m, conetop_coor, tune_factor):
        if cone_width_px > 0: #only updates when cone detected
            # position relative to the camera in meters.
            self.z_cor = cone_width_m * tune_factor / cone_width_px
            self.x_cor = conetop_coor[0] * self.z_cor / tune_factor
            self.y_cor = conetop_coor[1] * self.z_cor / tune_factor


    def rotate_coordinates(self, coor, x_angle, z_angle):
        rotation_matrix = np.array([[np.cos(x_angle), -1*np.sin(z_angle), 0],
                                    [np.cos(x_angle)*np.sin(z_angle), np.cos(x_angle)*np.cos(z_angle), -1*np.sin(x_angle)],
                                    [np.sin(x_angle)*np.sin(z_angle), np.sin(x_angle)*np.cos(z_angle), np.cos(x_angle)]])
        rotated_coor = rotation_matrix.dot(coor)
        #TODO Should have a return instead of assinment
        self.x_cor = rotated_coor[0]
        self.y_cor = rotated_coor[1]
        self.z_cor = rotated_coor[2]

    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint(self, image):
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv
        current_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        # Cone segmentation
        bin_im = self.get_cone_binary(current_image, treshold=80)

        #self.counter = self.counter + 1  # Counter to save picture under new name
        #if self.counter == 1:
        #   cv2.imwrite("all_pics/" + str(self.counter) + ".jpg", current_image)  # Write images to folder

        # Positioning in 2D of cone parts
        loc_2d = self.get_cone_2d_location(bin_im)
        # Get the position and width of the cone
        max_width = loc_2d[2]  # Max width detected, assumption biggest object is the cone.
        # Index of middle of max width
        # Remap 2D locations to 3D using width

        self.get_cone_3d_location(max_width, 0.18, [loc_2d[0], loc_2d[1]], 366)
        # tunefactor calculated by distance[m]*pixels of ob/seize obj[m]

    # Using the rotation angels of the camera to correct for the drone.

    def update_angles(self, angle):
        #update de angles and translation info
        #calculations will be done when service is called
        x = 0
        #self.rotate_coordinates(coor, x_angle=1, z_angle=1)
        #TODO update coordinates using these sensor values
        #print(angle.transforms)

    def handle_cor_req(self, req):
        #should also transform last coordinates
        print("Request received.")
        print([self.x_cor, self.y_cor, self.z_cor])
        return SendRelCorResponse(self.x_cor, self.y_cor, self.z_cor)

    #  Service for delivery of current relative coordinates
    def rel_cor_server(self):
        s = rospy.Service('rel_cor', SendRelCor, self.handle_cor_req)
        print("Waiting for request")

    def image_subscriber(self):
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.extract_waypoint)
        rospy.Subscriber("/tf", TFMessage, self.update_angles)

    def run(self):
        self.image_subscriber()
        self.rel_cor_server()
        rospy.spin()


if __name__ == "__main__":
    waypoint_extractor = WaypointExtractor()
    waypoint_extractor.run()
