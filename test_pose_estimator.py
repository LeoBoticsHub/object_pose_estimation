from pathlib import Path
from pose_estimator import PoseEstimator

yolact_weights = str(Path.home()) + "/Code/Vision/yolact/weights/yolact_plus_resnet50_54_800000.pth"
estimator = PoseEstimator(camera_type = 'REALSENSE',
                            obj_label = 'cup', 
                            obj_model_file = 'cup.ply', 
                            yolact_weights = yolact_weights, 
                            voxel_size = 0.0025)

filt_type = 'STATISTICAL'
filt_params_dict = {'nb_neighbors': 100, 'std_ratio': 0.1}

# filt_type = 'RADIUS'
# filt_params_dict = {'nb_points': 16, 'radius': 0.0025*2.5}

estimator.get_yolact_pcd(filt_type, filt_params_dict)








