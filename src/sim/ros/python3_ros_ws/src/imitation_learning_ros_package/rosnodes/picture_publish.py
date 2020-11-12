#!/usr/bin/env python
import numpy as np
import rospy
from sensor_msgs.msg import *
import cv2
class PicturePublisher:
    def __init__(self):
        path = r'src/sim/ros/python3_ros_ws/bagfiles/sample01.jpg'
        self.current_image = cv2.imread(path)

    def publish_pic(self):
        im_publisher = rospy.Publisher('image_topic', Image, queue_size=5)
        rospy.init_node('image_publisher', anonymous=True)
        rate = rospy.Rate(0.1)

        while not rospy.is_shutdown():
            image = Image()
            image.data = [int(5)]*300*600*3
            image.height = 600
            image.width = 300
            image.encoding = 'rgb8'
            rospy.loginfo(image)
            im_publisher.publish(image)
            rate.sleep()


if __name__ == '__main__':
    new_im_publisher = PicturePublisher()
    print("Publishing picture")
    new_im_publisher.publish_pic()
