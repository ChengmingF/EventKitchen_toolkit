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
        
        # #### Read DRGB -- RighttEvent extrinsic matrix
        # self.rotation_matrix_DRGB_RightEvent, self.translation_matrix_DRGB_RightEvent, _, _, _, _ = self.read_extrinsic_calibration(calibration_path, camA='DRGB', camB='RightEvent')
        
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

    def project_depth_to_event(
        self,
        depth0_mm: np.ndarray,    # (H0, W0) uint16 depth in mm, undistorted already, 0 invalid
        K0: np.ndarray,           # (3,3) intrinsics that MATCH the undistorted cam0 depth
        K1: np.ndarray,           # (3,3) intrinsics for event cam image (target)
        R_0to1: np.ndarray,       # (3,3) cam0 -> cam1
        T_0to1_mm: np.ndarray,    # (3,) or (3,1) cam0 -> cam1 translation in millimeters
        splat_kernel_size: int = 4,
    ):
        """
        Output:
        depth1_mm : (H1,W1) uint16 (0 invalid)
        Notes:
        - Uses a configurable target-pixel footprint to reduce projection holes
        - Uses per-pixel z-buffer so nearer points win (keeps occlusions correct)
        """
        W1, H1 = 1280, 720
        H0, W0 = depth0_mm.shape
        if splat_kernel_size < 1:
            raise ValueError("splat_kernel_size must be at least 1")

        # Convert depth to meters
        Z0 = depth0_mm.astype(np.float32) * 0.001
        valid = Z0 > 0
        if not np.any(valid):
            return np.zeros((H1, W1), dtype=np.uint16)

        # Pixel grid in cam0
        u0 = np.arange(W0, dtype=np.float32)
        v0 = np.arange(H0, dtype=np.float32)
        uu0, vv0 = np.meshgrid(u0, v0)

        uu0 = uu0[valid]
        vv0 = vv0[valid]
        z0 = Z0[valid]

        # Backproject (cam0) in meters
        fx0, fy0 = K0[0, 0], K0[1, 1]
        cx0, cy0 = K0[0, 2], K0[1, 2]
        X0 = (uu0 - cx0) * z0 / fx0
        Y0 = (vv0 - cy0) * z0 / fy0

        P0 = np.stack([X0, Y0, z0], axis=0)  # (3,N)

        # Transform to cam1
        R = R_0to1.astype(np.float32)
        T = T_0to1_mm.reshape(3, 1).astype(np.float32) * 0.001
        P1 = R @ P0 + T

        X1, Y1, Z1 = P1[0], P1[1], P1[2]
        infront = Z1 > 0
        if not np.any(infront):
            return np.zeros((H1, W1), dtype=np.uint16)

        X1, Y1, Z1 = X1[infront], Y1[infront], Z1[infront]

        # Project to cam1 (float pixel coords)
        fx1, fy1 = K1[0, 0], K1[1, 1]
        cx1, cy1 = K1[0, 2], K1[1, 2]
        u = fx1 * (X1 / Z1) + cx1
        v = fy1 * (Y1 / Z1) + cy1

        # Expand each projected point over a local footprint. This fills small
        # sampling gaps while the z-buffer preserves the nearest surface.
        zbuf = np.full((H1, W1), np.inf, dtype=np.float32)
        half_extent = (splat_kernel_size - 1) / 2.0
        x_start = np.floor(u - half_extent).astype(np.int32)
        y_start = np.floor(v - half_extent).astype(np.int32)

        for offset_y in range(splat_kernel_size):
            yi = y_start + offset_y
            for offset_x in range(splat_kernel_size):
                xi = x_start + offset_x
                inside = (xi >= 0) & (xi < W1) & (yi >= 0) & (yi < H1)
                np.minimum.at(zbuf, (yi[inside], xi[inside]), Z1[inside])

        # Depth output from z-buffer (meters -> mm)
        depth1_mm = np.zeros((H1, W1), dtype=np.uint16)
        depth_valid = zbuf < np.inf
        depth1_mm[depth_valid] = np.clip(
            zbuf[depth_valid] * 1000.0, 0, 65535
        ).astype(np.uint16)

        return depth1_mm
