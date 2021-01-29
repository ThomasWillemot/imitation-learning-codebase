#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import *
import time

class CameraVisualiser:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.init_node('camera_visualiser', anonymous=True)

    def show_image(self,image):
        cv_im = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        cv_im = np.ones((400, 600, 3))*255
        cv2.imshow("Image window", cv_im)
        cv2.waitKey(3)
        print('Where is the plot e')
    def run(self):
        rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.show_image)
        rospy.spin()
if __name__ == '__main__':
    new_wp_client = CameraVisualiser()
    print("Start showing data")
    new_wp_client.run()
    print("Done")