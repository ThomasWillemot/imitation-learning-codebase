#!/usr/bin/env python
import numpy as np
import rospy
import time as ti
from imitation_learning_ros_package.srv import SendRelCor


class WaypointClient:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

    def run_test(self):
        rospy.wait_for_service('rel_cor')
        test_run = 0
        while test_run < 3:
            try:
                get_rel_cor = rospy.ServiceProxy('rel_cor', SendRelCor)
                resp1 = get_rel_cor()
                print([resp1.x, resp1.y, resp1.z])
            except rospy.ServiceException as e:
                print("Service call failed: %s" % e)
            ti.sleep(2)
            test_run += 1

if __name__ == '__main__':
    new_wp_client = WaypointClient()
    print("Requesting COOR")
    new_wp_client.run_test()
    print("Test succeeded")
