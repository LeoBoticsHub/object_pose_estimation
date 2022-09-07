from pathlib import Path
from object_pose_estimation.pose_estimator import PoseEstimator
import open3d as o3d
import numpy as np
import copy

# yolact_weights = str(Path.home()) + "/Code/Vision/yolact/weights/yolact_plus_resnet50_54_800000.pth"
yolact_weights = str(Path.home()) + "/Code/Vision/yolact/weights/yolact_plus_resnet50_drill_74_750.pth"

# model_path = str(Path.home()) + "/Code/Vision/object_pose_estimation/test/models/mouse.ply"
model_path = str(Path.home()) + "/Code/Vision/object_pose_estimation/test/models/drill.ply"

estimator = PoseEstimator(camera_type = 'REALSENSE',            # Camera employed ('REALSENSE' or 'ZED')
                          obj_label = 'drill',                  # Yolact label
                          obj_model_path = model_path,          # Path to the PCD model
                          yolact_weights = yolact_weights,      # Path to Yolact weights
                          voxel_size = 0.0025,                  # Voxel size for downsamping
                          flg_plot = False)                     # Set to True to show intermidiate results

# Select PCD filtering method
filt_type = 'None'
filt_params_dict = {'nb_neighbors': 50, 'std_ratio': 0.2}
# filt_type = 'RADIUS'
# filt_params_dict = {'nb_points': 16, 'radius': 0.0025*2.5}

# Get PCD from Yolact-masked RGBD
obs_pcd, scene_pcd = estimator.get_yolact_pcd(filt_type, filt_params_dict)
obs_pcd.paint_uniform_color([0, 0.651, 0.929])
p = obs_pcd.get_center()
T_gl = estimator.global_registration(obs_pcd)

# Apply Global Registration
model_glob = copy.deepcopy(estimator.model_pcd).transform(T_gl)
model_glob.paint_uniform_color([1, 0.706, 0])

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
model_glob_frame = copy.deepcopy(world_frame).transform(T_gl)
o3d.visualization.draw_geometries([model_glob, obs_pcd, world_frame, model_glob_frame, scene_pcd], window_name = 'Global registration')

# Apply Local Registration via ICP
T_icp = estimator.local_registration(obs_pcd, T_gl, max_iteration = 100000, threshold = 0.2, method = 'p2p')

model_icp = copy.deepcopy(estimator.model_pcd).transform(T_icp)
model_icp.paint_uniform_color([1, 0.706, 0])
model_icp_frame = copy.deepcopy(world_frame).transform(T_icp)
o3d.visualization.draw_geometries([model_icp, obs_pcd, world_frame, model_icp_frame, scene_pcd], window_name = 'ICP local registration')

