import unittest
import src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.gate_waypoint_extractor as gwex
import unittest
import numpy as np
import rospy
import cv2
import matplotlib.pyplot as plt

class TestGateExtractor(unittest.TestCase):
    # Test the 3d localisation using proportional width to estimate depth
    def test_3d_localisation_width(self):
        fig = plt.figure(figsize=(848, 800))
        gate_extr = gwex.GateWaypointExtractor()
        path = 'gate_pics/1.jpg'
        current_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        bin_im = gate_extr.get_gate_binary(current_image, threshold=90)
        im_coor = gate_extr.get_gate_2d_speedup(bin_im)
        print(im_coor)
        print(gate_extr.get_cone_3d_location(im_coor[2], 0.18, [im_coor[0], im_coor[1]], 366))
        im_coor_plot = np.zeros((848, 800))
        im_coor_plot[-im_coor[1] + 400 - 1:-im_coor[1] + 400 + 2,
        im_coor[0] + 424:im_coor[0] + 424 + im_coor[2]] = 255 * np.ones((3, im_coor[2]))
        fig.add_subplot(1, 3, 1)
        plt.imshow(current_image, cmap='gray')
        fig.add_subplot(1, 3, 2)
        plt.imshow(bin_im, cmap='gray')
        fig.add_subplot(1, 3, 3)
        plt.imshow(im_coor_plot, cmap='gray')
        plt.show()

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
