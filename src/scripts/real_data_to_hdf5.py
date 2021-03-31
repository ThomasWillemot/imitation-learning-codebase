#!/usr/bin/python3.8
from src.core.data_types import Experience, TerminationType
from src.data.data_saver import DataSaver, DataSaverConfig
import os
import cv2
import numpy as np

# Creates binary training data in hdf5 format out of images.
class Real_data_to_hdf5:

    def __init__(self, image_folder, save_location, is_rectified):
        self.image_folder = image_folder
        self.image_count = 0
        self.output_dir = f'/media/thomas/Elements/experimental_data/real_world_data/{save_location}'
        os.makedirs(self.output_dir, exist_ok=True)
        config_dict = {
            'output_path': self.output_dir,
            'separate_raw_data_runs': True,
            'store_hdf5': True,
            'training_validation_split': 0 #this is only a validation set
        }
        # Create data saver
        config_data_saver = DataSaverConfig().create(config_dict=config_dict)
        self._data_saver = DataSaver(config=config_data_saver)
        self.is_rectified = is_rectified

    def run(self):
        directory = self.image_folder
        for filename in os.listdir(directory):
            if self.image_count < 20:
                if '1' in filename.split('_')[1]:
                    image_path = os.path.join(directory, filename)
                    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    processed_image = self.preprocess_image(grayscale_image)
                    self.save_experience_single_cone(processed_image, filename)
        self.save_hdf5()

    def preprocess_image(self,raw_image):
        if not self.is_rectified:
            dim = (848, 800)
            k = np.array(
                [[285.95001220703125, 0.0, 418.948486328125], [0.0, 286.0592956542969, 405.756103515625],
                 [0.0, 0.0, 1.0]])
            d = np.array(
                [[-0.006003059912472963], [0.04132957011461258], [-0.038822319358587265], [0.006561396177858114]])
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), k, dim,
                                                             cv2.CV_16SC2)
            raw_image = cv2.remap(raw_image, map1, map2, interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
        th, bin_im = cv2.threshold(raw_image, 150, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
        row_sum = np.sum(bin_im, axis=1)
        for row_idx in range(799):
            if row_sum[row_idx] > 400 * 255:
                airrow = row_idx
        bin_im[0:airrow, :] = 0
        i = airrow
        prev_empty = False
        while i < 799:
            curr_empty = row_sum[i] > 255
            if curr_empty:
                bin_im[i, :] = 0
            elif prev_empty:
                break
            else:
                prev_empty = True
            i += 1
        mask_path = '/media/thomas/Elements/Thesis/frame_drone_mask.png'
        mask = cv2.imread(mask_path, 0)
        img_masked = cv2.bitwise_and(bin_im, mask)
        return img_masked

    def save_experience_single_cone(self, image_np, file_name, action=None):
        experience = Experience()
        experience.done = TerminationType.NotDone
        if action is not None:
            experience.action = action
        else:
            experience.action = [0, 0, 0]
        experience.time_stamp = self.image_count
        experience.info = {"file_name": np.array([int(file_name.split('_')[0])])}
        experience.observation = image_np
        self._data_saver.save(experience)
        self.image_count += 1

    def save_hdf5(self):
        self._data_saver.create_train_validation_hdf5_files()
if __name__ == "__main__":
    input_path = '/media/thomas/Elements/Thesis_temp_data/pics/on_drone_rec_2'
    output_path = 'on_drone_no_frame'
    r2hdf5 = Real_data_to_hdf5(input_path,output_path, is_rectified=False)
    r2hdf5.run()
