import os
import shutil
import time
import unittest
from typing import List, Any, Union

import numpy as np
import rospy
from imitation_learning_ros_package.srv import SendRelCor
from src.sim.ros.src.process_wrappers import RosWrapper
import src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.waypoint_extractor as wpx
import cv2
import matplotlib.pyplot as plt
import time

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

    def test_3d_localisation_width(self):
        fig = plt.figure(figsize=(848,800))
        waypoint_extr = wpx.WaypointExtractor()
        path = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/17_2.jpg'
        current_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        bin_im = waypoint_extr.get_cone_binary(current_image, threshold=90)
        im_coor = waypoint_extr.get_cone_2d_location(bin_im)
        print(im_coor)
        print(waypoint_extr.get_cone_3d_location(im_coor[2], 0.18, [im_coor[0], im_coor[1]], 366))
        im_coor_plot = np.zeros((848,800))
        im_coor_plot[-im_coor[1]+400-1:-im_coor[1]+400+2, im_coor[0]+424:im_coor[0]+424+im_coor[2]] = 255*np.ones((3,im_coor[2]))
        fig.add_subplot(1, 3, 1)
        plt.imshow(current_image, cmap='gray')
        fig.add_subplot(1, 3, 2)
        plt.imshow(bin_im, cmap='gray')
        fig.add_subplot(1, 3, 3)
        plt.imshow(im_coor_plot, cmap='gray')
        plt.show()

    def test_3d_localisaton_triangulation(self):
        am_pics = 20
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        waypoint_extr = wpx.WaypointExtractor()

        dual_pictures = np.linspace(1,am_pics,am_pics)
        global_coor = []

        translation_matrix = self.get_translations()
        rot_m = self.get_rotations()
        start_time = time.time()
        for img_idx in range(len(dual_pictures)):
            path_1 = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/' + str(int(dual_pictures[img_idx])) + '_1.jpg'
            path_2 = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/' + str(int(dual_pictures[img_idx])) + '_2.jpg'
            current_image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
            current_image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)
            bin_im_1 = waypoint_extr.get_cone_binary(current_image_1, threshold=90)
            bin_im_2 = waypoint_extr.get_cone_binary(current_image_2, threshold=90)
            im_coor_1 = waypoint_extr.get_cone_2d_location(bin_im_1)
            im_coor_2 = waypoint_extr.get_cone_2d_location(bin_im_2)
            current_3d_coor = np.array(waypoint_extr.get_depth_triang(im_coor_1[0], im_coor_2[0], im_coor_1[1], im_coor_2[1]))
            #Rotation matrix
            angles = self.get_euler_angles(rot_m[img_idx])
            R = np.array([[np.cos(angles[0]), 0, np.sin(angles[0])],
                          [0, 0, 1],
                          [-1*np.sin(angles[0]), 0, np.cos(angles[0])]])
            #R = np.array([[1-2*(rot_m[img_idx][2]**2+rot_m[img_idx][3]**2),2*(rot_m[img_idx][1]*rot_m[img_idx][2]-rot_m[img_idx][0]*rot_m[img_idx][3]),2*(rot_m[img_idx][0]*rot_m[img_idx][2]-rot_m[img_idx][1]*rot_m[img_idx][3])],
            #    [2*(rot_m[img_idx][1]*rot_m[img_idx][2]-rot_m[img_idx][0]*rot_m[img_idx][3]),1-2*(rot_m[img_idx][1]**2+rot_m[img_idx][3]**2),2*(rot_m[img_idx][2]*rot_m[img_idx][3]-rot_m[img_idx][0]*rot_m[img_idx][1])],
            #    [2*(rot_m[img_idx][1]*rot_m[img_idx][3]-rot_m[img_idx][0]*rot_m[img_idx][2]),2*(rot_m[img_idx][0]*rot_m[img_idx][1]-rot_m[img_idx][2]*rot_m[img_idx][3]),1-2*(rot_m[img_idx][1]**2+rot_m[img_idx][2]**2)]])
            rotated_coor = R.dot(current_3d_coor)
            global_coor.append([rotated_coor[0]-translation_matrix[img_idx][0], rotated_coor[1]+translation_matrix[img_idx][1], rotated_coor[2]-translation_matrix[img_idx][2]])
            ax.scatter(global_coor[len(global_coor)-1][0], global_coor[len(global_coor)-1][1], global_coor[len(global_coor)-1][2])
        print("--- %s seconds ---" % (time.time() - start_time))
        with open('plot_3d_coor_global.txt', 'w') as f:
            for item in global_coor:
                f.write("%s\n" % item)
        plt.show()

    def get_euler_angles(self,quat):
        w = -quat[0]
        x = quat[2]
        y = quat[1]
        z = quat[3]
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.where(t2 > +1.0, +1.0, t2)
        # t2 = +1.0 if t2 > +1.0 else t2

        t2 = np.where(t2 < -1.0, -1.0, t2)
        # t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = np.arctan2(t3, t4)

        return [X, Y, Z]
    def get_translations(self):
        translations = []
        with open('transform_data/translation_data.txt') as f:
            lines = f.readlines()
            columns = []
            i = 0
            first_line = True
            for line in lines:
                line = line.strip()
                if line:
                    if first_line:
                        columns = [item.strip() for item in line.split(',')]
                        first_line = False
                    else:
                        if i == 20:
                            data = [item.strip() for item in line.split(',')]
                            translations.append([float(data[1]), float(data[2]), float(data[3])])
                            i = 0
                        if i == 7 or i == 14:  # depends on the frequency
                            data = [item.strip() for item in line.split(',')]
                            translations.append([float(data[1]), float(data[2]), float(data[3])])
                    i += 1
        return translations

    def get_rotations(self):
        rotations = []
        with open('transform_data/rotations.txt') as f:
            lines = f.readlines()
            columns = []
            i = 0
            first_line = True
            for line in lines:
                line = line.strip()
                if line:
                    if first_line:
                        columns = [item.strip() for item in line.split(',')]
                        first_line = False
                    else:
                        if i == 20:
                            data = [item.strip() for item in line.split(',')]
                            rotations.append([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
                            i = 0
                        if i == 7 or i == 14:  # depends on the frequency
                            data = [item.strip() for item in line.split(',')]
                            rotations.append([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
                    i = i+1
        return  rotations

    def start_test(self):
        #self.service_test()
        self.test_3d_localisaton_triangulation()

if __name__ == '__main__':
    unittest.main()
    print("Requesting COOR")
    print(start_test())
