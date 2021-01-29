
import time
import unittest
import numpy as np
import rospy
from cv_bridge import CvBridge
from imitation_learning_ros_package.srv import SendRelCor
import src.sim.ros.python3_ros_ws.src.handcrafted_cone_detection.src.waypoint_extractor as wpx
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#file to test functions of waypoint_extractor. Both testing the service and the coordinate extraction.
class TestWaypointExtractor(unittest.TestCase):

    # Test service, should return rel_cor
    def service_test(self):
        rospy.wait_for_service('rel_cor')
        try:
            get_rel_cor = rospy.ServiceProxy('rel_cor', SendRelCor)
            resp1 = get_rel_cor()
            return [resp1.x, resp1.y, resp1.z]
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    # Test multiple algorithms for extraction and get error
    def test_2d_extaction(self):
        nb_test_images = 50
        im_coors_errors_x = 0
        im_coors_errors_y = 0
        for i in range(nb_test_images):
            path_1 = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/' + str(i) +'_2.jpg'
            path_2 = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/' + str(i+1) + '_2.jpg'
            current_image = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
            second_image = cv2.imread(path_2,cv2.IMREAD_GRAYSCALE)
            waypoint_extr = wpx.WaypointExtractor()
            bin_im = waypoint_extr.get_cone_binary(current_image, threshold=90)
            bin_im_2 = waypoint_extr.get_cone_binary(second_image, threshold=90)
            im_coor_1 = waypoint_extr.get_cone_2d_location(bin_im)
            im_coor_2 = waypoint_extr.get_cone_2d_location(bin_im_2)
            start_time = time.time()
            im_coor_speed_1 = waypoint_extr.get_cone_2d_location_sum(bin_im)
            im_coor_speed_2 = waypoint_extr.get_cone_2d_location_sum(bin_im_2)
            print(time.time()-start_time)
            im_coors_errors_x += abs(im_coor_1[0]-im_coor_speed_1[0])
            im_coors_errors_x += abs(im_coor_2[0] - im_coor_speed_2[0])
            im_coors_errors_y += abs(im_coor_1[1] - im_coor_speed_1[1])
            im_coors_errors_y += abs(im_coor_2[1] - im_coor_speed_2[1])
            print('widtherro is %i' % (im_coor_1[2]-im_coor_speed_1[2]))
            print(i,im_coors_errors_x,im_coors_errors_y)
        print('pixel error for x_axis =  %f px' % (im_coors_errors_x / nb_test_images))
        print('pixel error for y_axis =  %f px' % (im_coors_errors_y / nb_test_images))

    # Test the 3d localisation using proportional width to estimate depth
    def test_3d_localisation_width(self):
        dim = (848, 800)
        k = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        d = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        pic_nr = 500
        fig = plt.figure(figsize=(848,800))
        waypoint_extr = wpx.WaypointExtractor()
        path = '/home/thomas/code/imitation-learning-codebase/on_drone_rec_2/' + str(pic_nr) + '_2.jpg'

        cv_im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rect_image = cv2.remap(cv_im, map1, map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
        bin_im = waypoint_extr.get_cone_binary(rect_image, threshold=150)
        bin_im[510:848, :] = np.zeros((290, 848))
        im_coor = waypoint_extr.get_cone_2d_location(bin_im)
        print(im_coor)
        #rel_coordin = waypoint_extr.get_cone_3d_location(im_coor[2], 0.18, [im_coor[0], im_coor[1]], 366)
        #print(rel_coordin)
        im_coor_plot = np.zeros((800,848))
        im_coor_plot[-im_coor[1]+400-1:-im_coor[1]+400+2, im_coor[0]+424:im_coor[0]+424+im_coor[2]] = 255*np.ones((3,im_coor[2]))
        fig.add_subplot(1, 3, 1)
        plt.imshow(rect_image, cmap='gray')
        fig.add_subplot(1, 3, 2)
        plt.imshow(bin_im, cmap='gray')
        fig.add_subplot(1, 3, 3)
        plt.imshow(im_coor_plot, cmap='gray')
        #plt.title(rel_coordin)
        plt.show()

    def test_new_possib(self):

        fig = plt.figure(figsize=(848, 800))
        waypoint_extr = wpx.WaypointExtractor()
        path = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/100_1.jpg'
        current_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        bin_im = waypoint_extr.get_cone_binary(current_image, threshold=90)
        im_coor = waypoint_extr.get_cone_2d_location(bin_im)
        fig.add_subplot(2, 3, 1)
        plt.imshow(current_image, cmap='gray')
        bin_im_filt = self.fast_2d_loc(bin_im)
        print(im_coor)
        rel_coordin = waypoint_extr.get_cone_3d_location(im_coor[2], 0.18, [im_coor[0], im_coor[1]], 366)
        print(rel_coordin)
        im_coor_plot = np.zeros((848, 800))
        im_coor_plot[-im_coor[1] + 400 - 1:-im_coor[1] + 400 + 2,
        im_coor[0] + 424:im_coor[0] + 424 + im_coor[2]] = 255 * np.ones((3, im_coor[2]))

        fig.add_subplot(2, 3, 2)
        plt.imshow(bin_im, cmap='gray')
        fig.add_subplot(2, 3, 3)
        plt.imshow(bin_im_filt, cmap='gray')
        fig.add_subplot(2, 3, 4)
        plt.imshow(im_coor_plot, cmap='gray')
        plt.title(rel_coordin)
        plt.show()
    def fast_2d_loc(self,bin_im):
        bin_im_filt = bin_im
        row_sum = np.sum(bin_im_filt, axis=1)
        i = 0
        while row_sum[i]>2:
            bin_im_filt[i,:] = np.zeros(800)
            i+=1
        col_sum = np.sum(bin_im_filt,axis=0)
        max_index_col = np.argmax(col_sum)
        max_index_row = np.argmax(np.sum(bin_im_filt,axis=1))
        print(max_index_col-400, -max_index_row+424,row_sum[max_index_row])
        return bin_im_filt


    def test_3d_localisation_trian_single_pic(self):
        dim = (848, 800)
        k = np.array(
            [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625], [0.0, 0.0, 1.0]])
        d = np.array(
            [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                                   cv2.CV_16SC2)
        im_nb = 2
        fig = plt.figure(figsize=(848,800))
        waypoint_extr = wpx.WaypointExtractor()
        '''
        path1 = 'all_fisheye_pics/'+str(im_nb)+'_1.jpg'
        path2 = 'all_fisheye_pics/'+str(im_nb)+'_2.jpg'
        '''
        path1 = 'on_drone_rec_2/' + str(im_nb) + '_1.jpg'
        path2 = 'on_drone_rec_2/' + str(im_nb) + '_2.jpg'
        current_image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        current_image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        start_time = time.time()
        rectified_image1 = cv2.remap(current_image1, map1, map2, interpolation=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        rectified_image2 = cv2.remap(current_image2, map1, map2, interpolation=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)

        bin_im1 = waypoint_extr.get_cone_binary(rectified_image1, threshold=150)
        bin_im2 = waypoint_extr.get_cone_binary(rectified_image2, threshold=150)
        bin_im1[510:848, :] = np.zeros((290, 848))
        bin_im2[510:848, :] = np.zeros((290, 848))
        im_coor1 = waypoint_extr.get_cone_2d_location(bin_im1)
        im_coor2 = waypoint_extr.get_cone_2d_location(bin_im2)

        print(im_coor1)
        print(im_coor2)

        rel_coordin_triang = waypoint_extr.get_depth_triang(im_coor1, im_coor2)
        print(time.time()-start_time)
        print(rel_coordin_triang)
        width = 848
        height = 800
        im_coor_plot1 = np.zeros((height, width))
        im_coor_plot2 = np.zeros((height, width))
        im_coor_plot1[int(-im_coor1[1]+height/2-3):int(-im_coor1[1]+height/2+2), int(im_coor1[0]-width/2-im_coor1[2]/2):int(im_coor1[0]-width/2+im_coor1[2]/2)]= 255*np.ones((5,im_coor1[2]))
        im_coor_plot2[int(-im_coor2[1] + height / 2 - 3):int(-im_coor2[1] + height / 2 + 2),
        int(im_coor2[0] - width / 2 - im_coor2[2] / 2):int(im_coor2[0] - width / 2 + im_coor2[2] / 2)] = 255 * np.ones(
            (5, im_coor2[2]))

        fig.add_subplot(2, 3, 1)
        plt.imshow(rectified_image1, cmap='gray')
        fig.add_subplot(2, 3, 2)
        plt.imshow(bin_im1, cmap='gray')
        fig.add_subplot(2, 3, 3)
        plt.imshow(im_coor_plot1, cmap='gray')
        fig.add_subplot(2, 3, 4)
        plt.imshow(rectified_image2, cmap='gray')
        fig.add_subplot(2, 3, 5)
        plt.imshow(bin_im2, cmap='gray')
        fig.add_subplot(2, 3, 6)
        plt.imshow(im_coor_plot2, cmap='gray')


        plt.show()


    def test_3d_localisaton_triangulation(self):
        am_pics = 50
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        waypoint_extr = wpx.WaypointExtractor()

        dual_pictures = np.linspace(1,am_pics,am_pics)
        global_coor = []
        im_coor_1 = [0, 0, 0]
        im_coor_2 = [0, 0, 0]
        translation_matrix = self.get_translations()
        rot_m = self.get_rotations()
        start_time = time.time()
        for img_idx in range(len(dual_pictures)):
            path_1 = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/' + str(int(dual_pictures[img_idx])) + '_1.jpg'
            path_2 = 'src/sim/ros/python3_ros_ws/src/imitation_learning_ros_package/rosnodes/all_pics/' + str(int(dual_pictures[img_idx])) + '_2.jpg'
            current_image_1 = cv2.imread(path_1, cv2.IMREAD_GRAYSCALE)
            current_image_2 = cv2.imread(path_2, cv2.IMREAD_GRAYSCALE)
            bin_im_1 = waypoint_extr.get_cone_binary(current_image_1, threshold=150)
            bin_im_2 = waypoint_extr.get_cone_binary(current_image_2, threshold=150)
            im_coor_1 = waypoint_extr.get_cone_2d_speedup(bin_im_1, im_coor_1)
            im_coor_2 = waypoint_extr.get_cone_2d_speedup(bin_im_2, im_coor_2)
            current_3d_coor = waypoint_extr.get_depth_triang(im_coor_1[0], im_coor_2[0], im_coor_1[1], im_coor_2[1])
            rotation_coordinates = np.array([current_3d_coor[0],current_3d_coor[1],current_3d_coor[2]])

            rot_mat = R.from_quat(rot_m[img_idx])
            translation = translation_matrix[img_idx+1]
            rotated_coor = self.transform_coordinates(rotation_coordinates, orientation=rot_mat.as_matrix(),translation=translation)
            global_coor.append(rotated_coor)
            ax.scatter(global_coor[rotated_coor[0], rotated_coor[1], rotated_coor[2]])
        print("--- %s seconds ---" % (time.time() - start_time))
        with open('plot_3d_coor_global.txt', 'w') as f:
            for item in global_coor:
                f.write("%s\n" % item)
        plt.show()

    def test_rotation_matrix(self):
        coordinate_1 = [1, -1, 0]
        coordinate_2 = np.array([1, 1, 0])
        quat = [0, 0, -0.7071068, 0.7071068]
        rot_mat = R.from_quat(quat)
        result =self.transform_coordinates(coordinate_2, rot_mat.as_matrix())
        self.assertAlmostEqual(coordinate_1[0], result[0], 5)
        self.assertAlmostEqual(coordinate_1[1], result[1], 5)
        self.assertAlmostEqual(coordinate_1[2], result[2], 5)

    def transform_coordinates(self, point: np.ndarray,
                  orientation: np.ndarray = np.eye(3),
                  translation: np.ndarray = np.zeros((3,)),
                  invert: bool = False) -> np.ndarray:
        transformation = np.zeros((4, 4))
        transformation[0:3, 0:3] = orientation
        transformation[0:3, 3] = translation
        transformation[3, 3] = 1
        point = np.concatenate([point, np.ones(1,)])
        if invert:
            transformation = np.linalg.inv(transformation)
        return np.matmul(transformation, point)


    def get_translations(self):
            translations = []
            with open('transform_data/translation_data.txt') as f:
                lines = f.readlines()
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
                                i = 0
                            if i == 1 or i == 15 or i == 8:  # depends on the frequency
                                data = [item.strip() for item in line.split(',')]
                                translations.append([float(data[1]), float(data[2]), float(data[3])])
                        i += 1
            return translations

    def get_rotations(self):
        rotations = []
        with open('transform_data/rotations.txt') as f:
            lines = f.readlines()
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
                            i = 0
                        if i == 1 or i == 15 or i == 8:  # depends on the frequency
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
