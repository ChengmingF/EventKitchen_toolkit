import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import torch 
import torch.nn as nn 
from glob import glob
from matplotlib import pyplot as plt


class Calibration_Loader():
    """
    Match pixels between event cameras and the dpeth caemra. 
    """
    def __init__(self, calibration_path):
        """
        Read intrinsic matrix
        """
        #### Read DRGB intrinsic matrix
        self.DRGB_intrinsic_matrix, self.DRGB_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'DRGB')

        #### Read LeftEvent intrinsic matrix
        self.LeftEvent_intrinsic_matrix, self.LeftEvent_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'LeftEvent')

        #### Read RightEvent intrinsic matrix
        self.RightEvent_intrinsic_matrix, self.RightEvent_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'RightEvent')
        
        # #### Read LeftRGB intrinsic matrix
        # self.LeftEvent_intrinsic_matrix, self.LeftEvent_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'LeftRGB')
        
        # #### Read RightRGB intrinsic matrix
        # self.LeftEvent_intrinsic_matrix, self.LeftEvent_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'RightRGB')
        
        """
        Read extrinsic matrix
        """
        #### Read LeftEvent -- DRGB extrinsic matrix
        self.rotation_matrix_LeftEvent_DRGB, self.translation_matrix_LeftEvent_DRGB, _, _, self.stereoMapDRGB_X, self.stereoMapDRGB_Y = self.read_extrinsic_calibration(calibration_path, camA='LeftEvent', camB='DRGB')

        #### Read LeftEvent -- RightEvent extrinsic matrix
        self.rotation_matrix_LeftEvent_RightEvent, self.translation_matrix_LeftEvent_RightEvent, self.stereoMapLeftEvent_X, self.stereoMapLeftEvent_Y, self.stereoMapRightEvent_X, self.stereoMapRightEvent_Y = self.read_extrinsic_calibration(calibration_path, camA='LeftEvent', camB='RightEvent')

        #### Read DRGB -- LeftEvent extrinsic matrix
        self.rotation_matrix_DRGB_LeftEvent, self.translation_matrix_DRGB_LeftEvent, _, _, _, _ = self.read_extrinsic_calibration(calibration_path, camA='DRGB', camB='LeftEvent')

        ####
        # add code to load the extrinsic matrix for LeftRGB and RightRGB if you need
        ####

    def read_intrinsic_calibration(self, calibration_path, cam):
        _intrinsic_file = os.path.join(calibration_path, f'calibration_results/intrinsic_calibration_results/{cam}_intrinsic.xml')
        assert os.path.isfile(_intrinsic_file), f'{_intrinsic_file} does not exist!'
        _intrinsic_loader = cv2.FileStorage(_intrinsic_file, cv2.FileStorage_READ)
        _intrinsic_matrix = _intrinsic_loader.getNode('Intrinsic_Matrix').mat()
        _distortion_matrix = _intrinsic_loader.getNode('Distortion_Matrix').mat()

        return _intrinsic_matrix, _distortion_matrix
    

    def read_extrinsic_calibration(self, calibration_path, camA, camB):
        _extrinsic_file = os.path.join(calibration_path, f'calibration_results/stereo_calibration_results/{camA}_{camB}_stereo_calibration.xml')
        assert os.path.isfile(_extrinsic_file), f'{_extrinsic_file} does not exist'
        _extrinsic_loader = cv2.FileStorage(_extrinsic_file, cv2.FileStorage_READ)
        _rotation_matrix = _extrinsic_loader.getNode('Rotation_Matrix').mat()
        _translation_matrix = _extrinsic_loader.getNode('Translation_Matrix').mat()
        _stereoMapA_X = _extrinsic_loader.getNode('stereoMapA_X').mat()
        _stereoMapA_Y = _extrinsic_loader.getNode('stereoMapA_Y').mat()
        _stereoMapB_X = _extrinsic_loader.getNode('stereoMapB_X').mat()
        _stereoMapB_Y = _extrinsic_loader.getNode('stereoMapB_Y').mat()

        return _rotation_matrix, _translation_matrix, _stereoMapA_X, _stereoMapA_Y, _stereoMapB_X, _stereoMapB_Y    


    def fast_reproject(self, depth_map, cameraB_instrinsic_matrix, translation_matrix, rotation_matrix):
        # create a meshgrid of pixel coordinates in DRGB (cam A)
        x_A, y_A = np.meshgrid(np.arange(1280), np.arange(720))

        # convert 2d coordiantes into normalized 3d
        Z = depth_map
        X = (x_A - self.DRGB_intrinsic_matrix[0, 2]) * Z / self.DRGB_intrinsic_matrix[0, 0]
        Y = (y_A - self.DRGB_intrinsic_matrix[1, 2]) * Z / self.DRGB_intrinsic_matrix[1, 1]

        # Stack X, Y, and Z to form 3D points in Camera A's coordinate frame
        points_3D_A = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

        # Apply rotation and translation to get points in Camera B's coordinate system
        points_3D_B = np.dot(rotation_matrix, points_3D_A.T) + translation_matrix.reshape(3, 1)
        points_3D_B = points_3D_B.T

        # Project 3D points in Camera B's coordinates into Camera B's image plane
        points_2D_B = np.dot(cameraB_instrinsic_matrix, points_3D_B.T)

        # Normalize to get pixel coordinates (divide by the Z coordinate)
        points_2D_B[0, :] /= points_2D_B[2, :]
        points_2D_B[1, :] /= points_2D_B[2, :]

        # Extract pixel coordinates in Camera B's image
        x_B = points_2D_B[0, :].reshape(720, 1280).astype(np.float32)
        y_B = points_2D_B[1, :].reshape(720, 1280).astype(np.float32)

        # Create a mask to identify valid pixels
        mask = (x_B >= 0) & (x_B < 1280) & (y_B >= 0) & (y_B < 720)
        
        # Create maps for remapping Camera A's frame to Camera B's image coordinates
        map_x_B = np.where(mask, x_B, 0)  # Only valid x_B coordinates
        map_y_B = np.where(mask, y_B, 0)  # Only valid y_B coordinates

        # Optionally, you can clip the maps to the valid range of Camera B's image size
        map_x_B = np.clip(map_x_B, 0, 1280 - 1)
        map_y_B = np.clip(map_y_B, 0, 720 - 1)

        # Warp Camera A's image to Camera B's perspective
        # Create an empty image for the projected output
        new_depth = np.zeros_like(depth_map)

        # Only warp the valid pixels from frame_A_undistorted to frame_A_warped_to_B
        new_depth[y_B[mask].astype(int), x_B[mask].astype(int)] = depth_map[y_A[mask].astype(int), x_A[mask].astype(int)]

        # post processing
        new_depth = self.interpolation_after_projection(new_depth)

        return new_depth


    def interpolation_after_projection(self, image):
        # convert to torch tensor
        image = torch.from_numpy(image).reshape(1, 1, 720, 1280).type(torch.float32)
        # maxpooling to vanish the 0 value in the projected image
        pool = nn.MaxPool2d(kernel_size=4, stride=1)
        image = pool(image)
        image = image.reshape(image.shape[2], image.shape[3])
        # to array
        image = np.array(image)
        # resize back to (1280, 720)
        projected_frame = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)

        return projected_frame


    def undistort_image(self, img, intinsic_matrix, distortion_matrix):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intinsic_matrix,
                                                          distortion_matrix,
                                                          (1280,720),
                                                          1,
                                                          (1280,720))
        dst = cv2.undistort(img, intinsic_matrix, distortion_matrix, None, newcameramtx)

        return dst, newcameramtx


    def project_and_rectify(self, ts):
        """
        This function can return the rectified LeftEvent frames, RightEvent frames, and depth map based on the prefixed timestamp of LeftEvent
        """
        """
        read images
        """ 
        # LeftEvent
        LeftEvent_img = os.path.join(self.LeftEvent_data_path, f'{"%.6f"%ts}.png')
        LeftEvent_image = cv2.imread(LeftEvent_img)
        # synced(nearst) depth
        depth_ts = self.DEPTH_ts[np.where(np.abs(self.DEPTH_ts-ts) == np.abs(self.DEPTH_ts-ts).min())][0]
        depth_img = os.path.join(self.DEPTH_data_path, 
                                 f'{"%.6f"%depth_ts}.tif')
        depth_map = cv2.imread(depth_img, -1).astype(np.int16)
        # synced(nearst) RightEvent
        RightEvent_ts = self.RightEvent_ts[np.where(np.abs(self.RightEvent_ts-ts) == np.abs(self.RightEvent_ts-ts).min())][0]
        RightEvent_img = os.path.join(self.RightEvent_data_path, 
                                  f'{"%.6f"%RightEvent_ts}.png')
        
        RightEvent_image = cv2.imread(RightEvent_img)
  

        """
        project depth to the FOV of LeftEvent
        """
        # undistort image
        depth_map, self.DRGB_intrinsic_matrix = self.undistort_image(depth_map,
                                                                     self.DRGB_intrinsic_matrix,
                                                                     self.DRGB_distortion_matrix)
        LeftEvent_image, self.LeftEvent_intrinsic_matrix = self.undistort_image(LeftEvent_image,
                                                                          self.LeftEvent_intrinsic_matrix,
                                                                          self.LeftEvent_distortion_matrix)
        RightEvent_image, self.RightEvent_intrinsic_matrix = self.undistort_image(RightEvent_image,
                                                                          self.RightEvent_intrinsic_matrix,
                                                                          self.RightEvent_distortion_matrix)
        # project
        projected_depth = self.fast_reproject(depth_map=depth_map,
                                              cameraB_instrinsic_matrix=self.LeftEvent_intrinsic_matrix,
                                              translation_matrix=self.translation_matrix_DRGB_LeftEvent,
                                              rotation_matrix=self.rotation_matrix_DRGB_LeftEvent)

        """
        Rectify LeftEvent, RightEvent, and projected depth
        """
        # rectify images
        rectified_LeftEvent_image = cv2.remap(LeftEvent_image,
                                 self.stereoMapLeftEvent_X,
                                 self.stereoMapLeftEvent_Y,
                                 cv2.INTER_LINEAR)
        rectified_RightEvent_image = cv2.remap(RightEvent_image,
                                 self.stereoMapRightEvent_X,
                                 self.stereoMapRightEvent_Y,
                                 cv2.INTER_LINEAR)
        rectified_projected_depth = cv2.remap(projected_depth,
                                    self.stereoMapLeftEvent_X,
                                    self.stereoMapLeftEvent_Y,
                                    cv2.INTER_LINEAR)

        f = plt.figure()
        ax = f.add_subplot(1, 3, 1)
        ax.imshow(rectified_LeftEvent_image)
        ax.set_title(f'Rectified_LeftEvent - {ts}')

        ax = f.add_subplot(1, 3, 2)
        ax.imshow(rectified_RightEvent_image)
        ax.set_title(f'Rectified_LeftEvent - {RightEvent_ts}')

        ax = f.add_subplot(1, 3, 3)
        ax.imshow(rectified_projected_depth)
        ax.set_title(f'Rectified_DEPTH - {depth_ts}')

        plt.show()

        return rectified_LeftEvent_image, rectified_RightEvent_image, rectified_projected_depth

if __name__ == '__main__':
    
    pc = 'mac'
    disk = 'NSEK0'
    if pc == 'mac':
        disk = os.path.join('/Volumes/', disk)
    elif pc == 'dell':
        disk = os.path.join('/media/chengming/', disk)
    kitchen = 'K_FCM'
    activity = 'cereal_bowl'

    projector = Pixel_Projector(calibration_path=os.path.join(disk, kitchen, 'calibration'),
                                data_path=os.path.join(disk, kitchen, activity))
    
    rectified_LeftEvent_image, rectified_RightEvent_image, rectified_projected_depth = projector.project_and_rectify(1704555877.345131)




