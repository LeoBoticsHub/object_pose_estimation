import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
from pathlib import Path
#from object_pose_estimation.pose_estimator import PCD_Obj_Combined
from object_pose_estimation.pose_estimator import PCD_multiple_obj_YOLO
import open3d as o3d

# Select Yolact weights
yolo_weights="/home/cto/object_pose_estimation/weights/pallet_two_cameras.pt" 
#yolact_weights = "/home/azunino/Documents/yolact/weights/yolact_plus_resnet50_plenv_LEONARDO_pcd_comb_79_240.pth" 


# cameras_dict = {'': 'REALSENSE'} # for a single camera
# IntelRealsense SERIAL NUMBER:
# 023322061667 : D415
# 023322062736 : D415
# 049122251418 : D455

# 242422303242 : D455 new
# 243122302060 : D455 new

##242422303242 camera 0 on the right
##243122302060 camera 1 on the left
cameras_dict = {'242422303242': 'REALSENSE', '243122302060': 'REALSENSE'} 
#cameras_dict = {'243122302060': 'REALSENSE'} 

estimator = PCD_multiple_obj_YOLO(cameras_dict = cameras_dict,          # Cameras employed {'serial_number': ('REALSENSE' or 'ZED')}
                          obj_label = 'Box',                  # Yolact label to find
                          yolo_weights = yolo_weights,      # Path to Yolact weights
                          voxel_size = 0.005,                   # Voxel size for downsamping
                          chess_size = (9, 6),                  # Number of chessboard corners
                          chess_square_size = 58.5,               # Chessboard size lenght [mm]
                          calib_loops = 100,                    # Number of samples for average calibration
                          flg_cal_wait_key = False,             # Set the wait-key calibration mode
                          flg_plot = True)                     # Set to True to show intermidiate results


filt_obj_pcd = estimator.get_yolo_pcd()
for object_idx in range(len(filt_obj_pcd)):
    obj_pcd = filt_obj_pcd[object_idx]
    obj_pcd.paint_uniform_color([0, 0.651, 0.929])
    estimator.compute_centroid(obj_pcd, display=True)

