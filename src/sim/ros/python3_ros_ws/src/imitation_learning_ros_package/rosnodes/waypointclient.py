#!/usr/bin/env python
import numpy as np
import rospy
from handcrafted_cone_detection.srv import SendRelCor
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class WaypointClient:
    def __init__(self):
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []
        self.plot_time = []
        self.counter = 0
        self.sleeptime = 50 #ms to wait
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'r')
        self.ln2, = plt.plot([], [], 'g')
        self.ln3, = plt.plot([], [], 'b')
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(-2, 5)
        print('Waiting for service')
        rospy.wait_for_service('/waypoint_extractor_server/rel_cor')
        print('service found')
        self.get_rel_cor = rospy.ServiceProxy('waypoint_extractor_server/rel_cor', SendRelCor)

    #Retrieves coordinates and plots them
    def animate(self, not_used):
        try:
            resp1 = self.get_rel_cor()

            if len(self.plot_time) > 999:
                self.x_vals = self.x_vals[1:]
                self.y_vals = self.y_vals[1:]
                self.z_vals = self.z_vals[1:]
                self.plot_time = self.plot_time[1:]
            self.plot_time.append(self.counter)
            self.x_vals.append(resp1.x)
            self.y_vals.append(resp1.y)
            self.z_vals.append(resp1.z)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        self.ln.set_data(self.plot_time, self.x_vals)
        self.ln2.set_data(self.plot_time, self.y_vals)
        self.ln3.set_data(self.plot_time, self.z_vals)
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
