import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
from pathlib import Path
from object_pose_estimation.pose_estimator import PoseEstimator
import open3d as o3d

# Select Yolact weights
# yolact_weights = str(Path.home()) + "/Code/yolact/weights/yolact_plus_resnet50_54_800000.pth"  # Standard weights
yolact_weights = str(Path.home()) + "/Code/yolact/weights/yolact_plus_resnet50_drill_74_750.pth" # fine-tuning for drill

# Select PCD model
# model_path = str(Path.home()) + "/Code/object_pose_estimation/test/models/mouse.ply"
# model_path = str(Path.home()) + "/Code/object_pose_estimation/test/models/cup.ply"
model_path = str(Path.home()) + "/Code/object_pose_estimation/test/models/drill.ply"

# cameras_dict = {'': 'REALSENSE'} # for a single camera
# IntelRealsense SERIAL NUMBER:
# 023322061667 : D415
# 023322062736 : D415
# 049122251418 : D455
cameras_dict = {'049122251418': 'REALSENSE', '023322062736': 'REALSENSE'} 

estimator = PoseEstimator(cameras_dict = cameras_dict,          # Cameras employed {'serial_number': ('REALSENSE' or 'ZED')}
                          obj_label = 'drill',                  # Yolact label to find
                          obj_model_path = model_path,          # Path to the PCD model
                          yolact_weights = yolact_weights,      # Path to Yolact weights
                          voxel_size = 0.005,                   # Voxel size for downsamping
                          chess_size = (5, 4),                  # Number of chessboard corners
                          chess_square_size = 40,               # Chessboard size lenght [mm]
                          calib_loops = 6,                    # Number of samples for average calibration
                          flg_cal_wait_key = True,             # Set the wait-key calibration mode
                          flg_plot = False)                     # Set to True to show intermidiate results

# Select PCD filtering method
# filt_type = None
# filt_params_dict = None
filt_type = 'STATISTICAL'
filt_params_dict = {'nb_neighbors': 50, 'std_ratio': 0.2}
# filt_type = 'RADIUS'
# filt_params_dict = {'nb_points': 16, 'radius': 0.0025*2.5}

# Get PCD from Yolact-masked RGBD
obj_pcd, scene_pcd = estimator.get_yolact_pcd(filt_type, filt_params_dict, flg_volume_int = False)
obj_pcd.paint_uniform_color([0, 0.651, 0.929])
T_gl = estimator.global_registration(obj_pcd)

# Apply Global Registration
model_glob = copy.deepcopy(estimator.model_pcd).transform(T_gl)
model_glob.paint_uniform_color([1, 0.706, 0])

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
model_glob_frame = copy.deepcopy(world_frame).transform(T_gl)
o3d.visualization.draw_geometries([model_glob, obj_pcd, world_frame, model_glob_frame, scene_pcd], window_name = 'Global registration')

# Apply Local Registration via ICP
T_icp = estimator.local_registration(obj_pcd, T_gl, max_iteration = 100000, threshold = 0.2, method = 'p2p')

model_icp = copy.deepcopy(estimator.model_pcd).transform(T_icp)
model_icp.paint_uniform_color([1, 0.706, 0])
model_icp_frame = copy.deepcopy(world_frame).transform(T_icp)

model_icp_plot = copy.deepcopy(model_icp).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
obj_pcd_plot = copy.deepcopy(obj_pcd).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
world_frame_plot = copy.deepcopy(world_frame).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
model_icp_frame_plot = copy.deepcopy(model_icp_frame).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
scene_pcd_plot = copy.deepcopy(scene_pcd).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([model_icp_plot, obj_pcd_plot, world_frame_plot, model_icp_frame_plot, scene_pcd_plot], window_name = 'ICP local registration')

# To flip PCDs: pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
