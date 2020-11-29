import os
import shutil
import time
import unittest

import numpy as np
import rospy
from imitation_learning_ros_package.srv import SendRelCor
from src.sim.ros.src.process_wrappers import RosWrapper
import src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.waypoint_extractor as wpx
import cv2
import matplotlib.pyplot as plt

#file to test functions of waypoint_extractor. Both testing the service and the coordinate extraction.

class TestWaypointExtractor(unittest.TestCase):


    def service_test(self):
        rospy.wait_for_service('rel_cor')
        try:
            get_rel_cor = rospy.ServiceProxy('rel_cor', SendRelCor)
            resp1 = get_rel_cor()
            return [resp1.x, resp1.y, resp1.z]
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def test_2d_localisation(self):
        fig = plt.figure(figsize=(848,800))
        waypoint_extr = wpx.WaypointExtractor()
        path = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/1.jpg'
        current_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        bin_im = waypoint_extr.get_cone_binary(current_image, treshold=90)
        im_coor = waypoint_extr.get_cone_2d_location(bin_im)
        print(im_coor)
        print(waypoint_extr.get_cone_3d_location(im_coor[2], 0.18, [im_coor[0], im_coor[1]], 366))
        im_coor_plot = np.zeros((800,848))
        im_coor_plot[-im_coor[1]+400-1:-im_coor[1]+400+2, im_coor[0]+424:im_coor[0]+424+im_coor[2]] = 255*np.ones((3,im_coor[2]))
        fig.add_subplot(1, 3, 1)
        plt.imshow(current_image, cmap='gray')
        fig.add_subplot(1, 3, 2)
        plt.imshow(bin_im, cmap='gray')
        fig.add_subplot(1, 3, 3)
        plt.imshow(im_coor_plot, cmap='gray')
        plt.show()
    def start_test(self):
        #self.service_test()
        self.test_2d_localisation()

if __name__ == '__main__':
    unittest.main()
    print("Requesting COOR")
    print(start_test())
