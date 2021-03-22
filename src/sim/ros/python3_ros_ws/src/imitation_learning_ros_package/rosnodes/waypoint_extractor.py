#!/usr/bin/python3.8
"""Extract waypoints out of a single image.
    Has two functionolities
        Image retrieval: calculates 3d location based on an image
        Service for the controller which answers with a 3d location
"""
import numpy as np
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import TransformStamped, Transform, Vector3
from sensor_msgs.msg import *
from tf2_msgs.msg import *

sys.path.append('src/sim/ros/test')
from imitation_learning_ros_package.srv import SendRelCor, SendRelCorResponse
import cv2
from std_msgs.msg import *
from geometry_msgs import *


class WaypointExtractor:

    def __init__(self):
        self.bridge = CvBridge()
        dim = (800, 848)
        k = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        d = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        self.counter = 0
        self.x_1 = -1
        self.y_1 = -1
        self.x_2 = 0
        self.y_2 = 0
        self.x_orig = 0
        self.y_orig = 0
        self.z_orig = 0
        self.imagexite_rec = 0
        rospy.init_node('waypoint_extractor_server')

    # Function to extract the cone out of an image. The part of the cone(s) are binary ones, the other parts are 0.
    # inputs: image and color of cone
    # output: binary of cone
    def get_cone_binary(self, current_image, threshold):
        binary_image = cv2.threshold(current_image, threshold, 1, cv2.ADAPTIVE_THRESH_MEAN_C)
        return binary_image[1]

    # Extract the 2d location in the image after segmentation.
    # TODO: loops of get_cone_binary and this function can be written in one.
    # Extract the 2d location in the image after segmentation.
    # Input: binary/thersholed image
    # output: x(width in image), y(height in image), width of cone (in pixels)
    def get_cone_2d_location(self, bin_im):
        row_sum = np.sum(bin_im, axis=1)
        i = 0

        while row_sum[i] > 1 and i < 799:
            bin_im[i, :] = np.zeros(848)
            i += 1

        airrow = 0
        for row_idx in range(799):
            if row_sum[row_idx] > 400 * 255:
                airrow = row_idx
        bin_im[1:airrow, :] = 0
        row_sum = np.sum(bin_im, axis=1)
        cone_found = False
        cone_row = 0
        max_row = 0
        row = 799  # start where no drone parts are visible in image
        cone_started = False
        while not cone_found and row >= 0:
            if row_sum[row] >= max_row and row_sum[row] > 4 * 255:
                cone_row = row
                max_row = row_sum[row]
                cone_started = True
            elif cone_started:
                cone_found = True
            row -= 1

        current_start = 0
        max_start = 0
        max_width = 0
        current_width = 0
        for col_index in range(847):
            if bin_im[cone_row, col_index] == 0:
                if current_width > max_width:
                    max_width = current_width
                    max_start = current_start
                current_width = 0
                current_start = 0
            else:
                if current_start == 0:
                    current_start = col_index
                current_width += 1
        return [max_start + int(np.ceil(max_width / 2)) - 424, -cone_row + 400, max_width]

    # 3d coordinate estimation using
    def get_cone_3d_location(self, cone_width_px, cone_width_m, conetop_coor, tune_factor):
        x_cor = 0
        y_cor = 0
        z_cor = -1  # do not update if z remains -1 TODO
        if cone_width_px > 0:  # only updates when cone detected
            # position relative to the camera in meters.
            z_cor = cone_width_m * tune_factor / cone_width_px
            x_cor = conetop_coor[0] * z_cor / tune_factor
            y_cor = conetop_coor[1] * z_cor / tune_factor
        return [z_cor, -x_cor, y_cor]

    def rotate_coordinates(self, coor, x_angle, z_angle):
        rotation_matrix = np.array([[np.cos(x_angle), -1 * np.sin(z_angle), 0],
                                    [np.cos(x_angle) * np.sin(z_angle), np.cos(x_angle) * np.cos(z_angle),
                                     -1 * np.sin(x_angle)],
                                    [np.sin(x_angle) * np.sin(z_angle), np.sin(x_angle) * np.cos(z_angle),
                                     np.cos(x_angle)]])
        rotated_coor = rotation_matrix.dot(coor)
        # TODO Should have a return instead of assinment
        self.x_cor = rotated_coor[0]
        self.y_cor = rotated_coor[1]
        self.z_cor = rotated_coor[2]

    def get_depth_triang(self, x_fish1, x_fish2, y_fish1, y_fish2):
        baseline = 0.064  # 6.4mm
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
        current_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        path = 'gate_rec/'+str(self.counter) + '_1.jpg'
        cv2.imwrite(path, current_image)
        # Cone segmentation
        #bin_im = self.get_cone_binary(current_image, threshold=80)
        # Positioning in 2D of cone parts
        #loc_2d = self.get_cone_2d_location(bin_im)
        # Get the position and width of the cone
        #max_width = loc_2d[2]  # Max width detected, assumption biggest object is the cone.
        # Index of middle of max width
        # Remap 2D locations to 3D using width
        #self.x_1 = loc_2d[0]
        #self.y_1 = loc_2d[1]
        # [self.x_orig,self.y_orig,self.z_orig] = self.get_cone_3d_location(max_width, 0.18, [loc_2d[0], loc_2d[1]], 366)
        # tunefactor calculated by distance[m]*pixels of ob/seize obj[m]

    def extract_waypoint_2(self, image):
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')  # Load images to cv
        path = 'gate_rec/' + str(self.counter) + '_2.jpg'
        self.counter += 1
        cv2.imwrite(path, cv_im)

        current_image = cv2.remap(cv_im, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)  # Remap fisheye to normal picture
        cv2.imwrite(path, current_image)

        # Cone segmentation
        #bin_im = self.get_cone_binary(current_image, threshold=80)
        # Positioning in 2D of cone parts
        #loc_2d = self.get_cone_2d_location(bin_im)
        # Get the position and width of the cone
        #max_width = loc_2d[2]  # Max width detected, assumption biggest object is the cone.
        # Index of middle of max width
        # Remap 2D locations to 3D using width
        #self.x_2 = loc_2d[0]
        #self.y_2 = loc_2d[1]

    # Using the rotation angels of the camera to correct for the drone.
    def update_angles(self, data):
        # update de angles and translation info
        print(data.transforms[0].transform.rotation)
        # TODO update coordinates using these sensor values

    # Handles the service requests.
    def handle_cor_req(self, req):
        # should also transform last coordinates
        print("Request received.")
        coor = self.get_depth_triang(self.x_1, self.x_2, self.y_1, self.y_2)
        print('Triangulation: ')
        print([coor[0], coor[1], coor[2]])
        return SendRelCorResponse(coor[0], coor[1], coor[2])
        # return SendRelCorResponse(self.x_orig, self.y_orig, self.z_orig)

    #  Service for delivery of current relative coordinates
    def rel_cor_server(self):
        s = rospy.Service('rel_cor', SendRelCor, self.handle_cor_req)
        print("Waiting for request")

    # Subscribes to topics and and runs callbacks
    def image_subscriber(self):
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.extract_waypoint_1)
        rospy.Subscriber("/camera/fisheye2/image_raw", Image, self.extract_waypoint_2)
        #rospy.Subscriber("/tf", TFMessage, self.update_angles)

    # Starts all needed functionalities
    def run(self):
        self.image_subscriber()
        self.rel_cor_server()
        rospy.spin()


if __name__ == "__main__":
    waypoint_extractor = WaypointExtractor()
    waypoint_extractor.run()
