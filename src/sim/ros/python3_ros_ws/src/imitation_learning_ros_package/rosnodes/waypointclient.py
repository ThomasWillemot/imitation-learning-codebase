#!/usr/bin/env python
import numpy as np
import rospy
from imitation_learning_ros_package.srv import SendRelCor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class WaypointClient:
    def __init__(self):
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []
        self.counter = 0
        self.sleeptime = 1000 #ms to wait
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'ro')
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(0, 50)
        print('Waiting for service')
        rospy.wait_for_service('rel_cor')
        print('service found')
        self.get_rel_cor = rospy.ServiceProxy('rel_cor', SendRelCor)

    #Retrieves coordinates and plots them
    def animate(self, not_used):
        try:
            resp1 = self.get_rel_cor()
            self.x_vals.append(resp1.x)
            self.y_vals.append(resp1.y)
            self.z_vals.append(resp1.z)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        self.ln.set_data(self.x_vals, self.z_vals)
        self.counter += 1

    # Run the plotter using the data of the service.
    def run_test(self):

        ani = FuncAnimation(self.fig, self.animate, interval=self.sleeptime)  # Call service each sec for 3d loc and plot
        plt.show() #Needed to show plots


if __name__ == '__main__':
    new_wp_client = WaypointClient()
    print("Start plotting")
    new_wp_client.run_test()
    print("Done")
