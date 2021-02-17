#!/usr/bin/python3
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
from imitation_learning_ros_package.srv import SendRelCor, SendRelCorResponse
import cv2
from std_msgs.msg import *
from geometry_msgs import *


class GateWaypointExtractor:

    def __init__(self):
        self.bridge = CvBridge()
        dim = (800, 848)
        self.counter = 0
        self.x_1 = -1
        self.y_1 = -1
        self.x_2 = 0
        self.y_2 = 0
        self.x_orig = 0
        self.y_orig = 0
        self.z_orig = 0
        self.imagexite_rec = 0
        rospy.init_node('gate_extractor_server')

    # Function to extract the gate out of an image. The part of the gate(s) are binary ones, the other parts are 0.
    # inputs: BW image of cone
    # output: binary of cone
    def get_gate_binary(self, current_image, threshold):
        binary_image = cv2.threshold(current_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image[1]

    def get_gate_2d_speedup(self, bin_im, prev_coor):
        if prev_coor[2] == 0:
            return self.get_cone_2d_strided(bin_im)
        x_prev_pix = prev_coor[0] + 400
        y_prev_pix = -prev_coor[1] + 424
        if prev_coor[1] > 846 or prev_coor[0] <2 or prev_coor[1] <2 or prev_coor[0] >= 799:
            return self.get_cone_2d_strided(bin_im)
        return self.search_diag(bin_im, x_prev_pix, y_prev_pix)

    def search_left_up(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix + 1, x_prev_pix - 1] > 0:  # left
            return self.search_left_down(bin_im, x_prev_pix - 1, y_prev_pix + 1)
        elif bin_im[y_prev_pix - 1, x_prev_pix - 1] > 0:  # down
            return self.search_left_up(bin_im, x_prev_pix - 1, y_prev_pix - 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_right_up(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix + 1, x_prev_pix + 1] > 0:  # left
            return self.search_right_down(bin_im, x_prev_pix + 1, y_prev_pix + 1)
        elif bin_im[y_prev_pix - 1, x_prev_pix + 1] > 0:  # up
            return self.search_right_up(bin_im, x_prev_pix + 1, y_prev_pix - 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_left_down(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix - 1, x_prev_pix - 1] > 0:  # right
            return self.search_left_up(bin_im, x_prev_pix - 1, y_prev_pix - 1)
        elif bin_im[y_prev_pix + 1, x_prev_pix - 1] > 0:  # down
            return self.search_left_down(bin_im, x_prev_pix - 1, y_prev_pix + 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_right_down(self, bin_im, x_prev_pix, y_prev_pix):
        if y_prev_pix == 847 or x_prev_pix == 0 or y_prev_pix == 0 or x_prev_pix == 799:
            return [x_prev_pix, y_prev_pix]
        if bin_im[y_prev_pix - 1, x_prev_pix + 1] > 0:  # right
            return self.search_right_up(bin_im, x_prev_pix + 1, y_prev_pix - 1)
        elif bin_im[y_prev_pix + 1, x_prev_pix + 1] > 0:  # up
            return self.search_right_down(bin_im, x_prev_pix + 1, y_prev_pix + 1)
        else:
            return [x_prev_pix, y_prev_pix]

    def search_diag(self, bin_im, prev_pix_x, prev_pix_y):
        # up left
        left_coor = [0, 0]
        right_coor = [0, 0]
        up_coor = [0, 0]
        down_coor = [0, 0]
        if bin_im[prev_pix_y - 1, prev_pix_x - 1]:
            up_coor = self.search_left_up(bin_im, prev_pix_x - 1, prev_pix_y - 1)
        # right up
        if bin_im[prev_pix_y - 1, prev_pix_x + 1]:
            right_coor = self.search_right_up(bin_im, prev_pix_x + 1, prev_pix_y - 1)
        # left down
        if bin_im[prev_pix_y + 1, prev_pix_x - 1]:
            left_coor = self.search_right_down(bin_im, prev_pix_x - 1, prev_pix_y + 1)
        # down right
        if bin_im[prev_pix_y + 1, prev_pix_x + 1]:
            down_coor = self.search_right_down(bin_im, prev_pix_x + 1, prev_pix_y + 1)
        left_side_obj = min(left_coor[0], up_coor[0])
        top_side_obj = min(left_coor[1], up_coor[1])
        right_side_obj = max(down_coor[0], right_coor[0])
        down_side_obj = max(down_coor[1], right_coor[1])

        if left_side_obj == right_side_obj:
            return self.get_cone_2d_strided(bin_im)
        else:
            return [int(np.ceil((left_side_obj + right_side_obj) / 2) - 400),
                    int(-1 * (np.ceil(top_side_obj + down_side_obj) / 2) + 424),
                    right_side_obj - left_side_obj]

    def get_gate_2d_strided(self, bin_im):
        gate_found = False
        im_size = bin_im.shape
        max_width_row = 0
        max_width_x = -1
        max_width_y = -1
        current_width_start = -1
        current_width = 0
        prev_pix = 0
        row = im_size[0] - 1
        prev_row = 0
        while not gate_found and row >= 0:
            for column in range(int(im_size[1] / 2)):
                if bin_im[row, column * 2] > 0:
                    if current_width_start == -1:
                        current_width_start = column * 2
                    elif prev_pix == 1:
                        current_width += 1
                    elif current_width == 0:
                        current_width_start = column * 2
                    prev_pix = 1
                else:
                    prev_pix = 0
            if current_width > max_width_row and current_width_start > 0:
                max_width_row = current_width
                max_width_x = current_width_start
                max_width_y = row
            if prev_row == 1 and current_width_start == -1 and max_width_row > 2:
                cone_found = True
            if current_width_start > -1:
                prev_row = 1
            current_width = 0
            current_width_start = -1
            row -= 2
        max_width_row = max_width_row * 2

        return [max_width_x - 400 + int(np.ceil(max_width_row / 2)), -max_width_y + 424,
                max_width_row]  # counting starts at zero

    def get_depth_triang(self, x_fish1, x_fish2, y_fish1, y_fish2):
        baseline = 0.064  # 6.4mm???
        disparity = x_fish1 - x_fish2
        if disparity == 0:
            disparity = 1
        x = baseline * x_fish1 / disparity
        y = baseline * y_fish1 / disparity
        z = baseline * 286 / disparity
        return [z, -x, y]

    # Extracts the waypoints (3d location) out of the current image.
    def extract_waypoint_1(self, image):
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv
        path = 'gate_pics/'+str(self.counter) + '.jpg'
        cv2.imwrite(path, cv_im)

    # Subscribes to topics and and runs callbacks
    def image_subscriber(self):
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.extract_waypoint_1)

    # Starts all needed functionalities
    def run(self):
        self.image_subscriber()
        rospy.spin()


if __name__ == "__main__":
    waypoint_extractor = GateWaypointExtractor()
    waypoint_extractor.run()
