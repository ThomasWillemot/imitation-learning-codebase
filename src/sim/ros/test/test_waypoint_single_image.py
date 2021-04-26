import unittest
import cv2
import src.sim.ros.python3_ros_ws.src.imitation_learning_ros_package.rosnodes.waypoint_extractor as wpx

class Test_waypoint_single_image(unittest.TestCase):
    def test_data_loader(self):
        path_actions = f'/media/thomas/Elements/experimental_data/binary_test_set_800/data_collection_gazebo/raw_data/21-03-13_11-32-57/action.data'
        action_docu = open(path_actions, "r")
        lines = action_docu.readlines()
        nb_lines = len(lines)
        mse_sum = 0
        abs_depth_error_sum = 0
        for line_nb in range(nb_lines):
            line = lines[line_nb]
            line = line.strip("\n")
            splitted_line = line.split(" ")
            x = float(splitted_line[2])
            y = float(splitted_line[3])
            z = float(splitted_line[4])
            zfill_image_nb = str(line_nb).zfill(15)
            path = f'/media/thomas/Elements/experimental_data/binary_test_set_800/data_collection_gazebo/raw_data/21-03-13_11-32-57/observation/'+zfill_image_nb+'.jpg'
            binary_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            waypoint_ext = wpx.WaypointExtractor()
            im_coor_cone = waypoint_ext.get_cone_2d_location(binary_image)
            coor_cone = waypoint_ext.get_cone_3d_location(im_coor_cone[2], 0.18, [im_coor_cone[0], im_coor_cone[1]], 447)
            mse_sum += 1/3*((x-coor_cone[0])**2+(y-coor_cone[1])**2+(z-coor_cone[2])**2)
            abs_depth_error_sum += abs(x-coor_cone[0])
        mse = mse_sum/nb_lines
        abs_dept_error = abs_depth_error_sum/nb_lines
        print('MSE = ' + "{:.2f}".format(mse))
        print('Abs error x = ' "{:.2f}".format(abs_dept_error))
if __name__ == '__main__':
    unittest.main()
