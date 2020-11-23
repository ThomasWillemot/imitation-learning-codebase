#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import *
import cv2
class PicturePublisher:
    def __init__(self):
        self.current_image = []
        self.height = 0
        self.width = 0

    def publish_pic(self):
        im_publisher = rospy.Publisher('image_topic', Image, queue_size=5)
        rospy.init_node('image_publisher', anonymous=True)
        rate = rospy.Rate(0.05)

        while not rospy.is_shutdown():
            path = r'src/sim/ros/python3_ros_ws/bagfiles/sample01.jpg'
            image_format = cv2.imread(path)
            image_size = image_format.shape
            r_image = image_format[:,:,2]
            print(image_size)
            self.height = image_size[0]
            self.width = image_size[1]
            self.current_image = r_image.tolist()
            print(len(self.current_image))
            print(self.current_image[0])
            image = Image()
            image.data = self.current_image
            image.height = self.height
            image.width = self.width
            image.encoding = 'rgb8'
            im_publisher.publish(image)
            print("Sending...")
            rate.sleep()



if __name__ == '__main__':
    new_im_publisher = PicturePublisher()
    print("Publishing picture")
    new_im_publisher.publish_pic()
