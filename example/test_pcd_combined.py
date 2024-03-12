import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
from pathlib import Path
#from object_pose_estimation.pose_estimator import PCD_Obj_Combined
from object_pose_estimation.pose_estimator import PCD_Obj_Combined_YOLO
import open3d as o3d

# Select Yolact weights
yolo_weights="/home/azunino/Downloads/Data_Robotic_Arms/runs/segment/train4/weights/best.pt" 
#yolact_weights = "/home/azunino/Documents/yolact/weights/yolact_plus_resnet50_plenv_LEONARDO_pcd_comb_79_240.pth" 


# cameras_dict = {'': 'REALSENSE'} # for a single camera
# IntelRealsense SERIAL NUMBER:
# 023322061667 : D415
# 023322062736 : D415
# 049122251418 : D455

# 242422303242 : D455 new
# 243122302060 : D455 new
     
cameras_dict = {'243122302060': 'REALSENSE', '242422303242': 'REALSENSE'} 
#cameras_dict = {'243122302060': 'REALSENSE'} 

estimator = PCD_Obj_Combined_YOLO(cameras_dict = cameras_dict,          # Cameras employed {'serial_number': ('REALSENSE' or 'ZED')}
                          obj_label = 'plenv',                  # Yolact label to find
                          yolo_weights = yolo_weights,      # Path to Yolact weights
                          voxel_size = 0.005,                   # Voxel size for downsamping
                          chess_size = (5, 4),                  # Number of chessboard corners
                          chess_square_size = 40,               # Chessboard size lenght [mm]
                          calib_loops = 6,                    # Number of samples for average calibration
                          flg_cal_wait_key = True,             # Set the wait-key calibration mode
                          flg_plot = True)                     # Set to True to show intermidiate results

# Select PCD filtering method
# filt_type = None
# filt_params_dict = None
filt_type = 'STATISTICAL'
filt_params_dict = {'nb_neighbors': 50, 'std_ratio': 0.2}
# filt_type = 'RADIUS'
# filt_params_dict = {'nb_points': 16, 'radius': 0.0025*2.5}

# Get PCD from Yolact-masked RGBD
#obj_pcd, scene_pcd = estimator.get_yolact_pcd(filt_type, filt_params_dict, flg_volume_int = False)
obj_pcd, scene_pcd = estimator.get_yolo_pcd(filt_type, filt_params_dict, flg_volume_int = False)
obj_pcd.paint_uniform_color([0, 0.651, 0.929])
estimator.compute_centroid(obj_pcd, display=True)
