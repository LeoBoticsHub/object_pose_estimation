import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy

from pathlib import Path
yolact_path = str(Path.home()) + '/Code/Vision/yolact'
sys.path.append(yolact_path)

camera_path = str(Path.home()) + '/Code/Vision/camera_utils/src'
sys.path.append(camera_path)

ai_path = str(Path.home()) + '/Code/Vision/ai_utils/src'
sys.path.append(ai_path)

from camera_utils.camera_init import IntelRealsense
from camera_utils.camera_init import Zed
from ai_utils.YolactInference import YolactInference

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name = 'Filtering')

class PoseEstimator:
    def __init__(self, camera_type, obj_label, obj_model_file, yolact_weights, voxel_size):
        try:
            self.yolact = YolactInference(model_weights=yolact_weights, display_img = False)
        except:
            raise ValueError('Yolact inizialization error')

        if camera_type == 'REALSENSE':
            self.camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD)
        elif camera_type == 'ZED':
            self.camera = Zed(rgb_resolution=Zed.Resolution.HD)
        else:
            sys.exit("Wrong camera type!")

        print("Camera initialization")
        for i in range(30):
            _, _ = self.camera.get_aligned_frames()

        self.obj_label = obj_label      # object yolact label
        self.voxel_size = voxel_size    # downsampling voxel size

        print("Load object model")
        try:
            self.model_pcd = o3d.io.read_point_cloud('model/'+obj_model_file)
        except:
            raise ValueError('Error loading object model')
        
        self.model_pcd = self.model_pcd.voxel_down_sample(self.voxel_size) # 1. Points are bucketed into voxels.
                                                                           # 2. Each occupied voxel generates exact one point by averaging all points inside.
    


    def get_yolact_pcd(self, filt_type, filt_params_dict):
        print("Get camera frames")
        rgb_frame, depth_frame = self.camera.get_aligned_frames()
        rgb_frame = np.array(rgb_frame)
        rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth_frame.astype(np.uint16)))

        # set intrinsics for open3d
        width = max(depth_frame.shape[0], depth_frame.shape[1])
        height = min(depth_frame.shape[0], depth_frame.shape[1])
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, self.camera.intr['fx'], self.camera.intr['fy'], self.camera.intr['px'], self.camera.intr['py'])
            
        print("Yolact inference")
        infer = self.yolact.img_inference(rgb_frame, classes=[self.obj_label])

        if len(infer) != 0:
            boxes = infer[self.obj_label]['boxes']
            masks = infer[self.obj_label]['masks']
            if len(boxes) == 1:
                
                rgb_frame_new = rgb_frame.copy()
                depth_frame_new = depth_frame.copy()
                depth_frame_new = np.array(depth_frame_new * masks[0], dtype = np.uint16)
                
                for i in range(3):
                    rgb_frame_new[:,:,i] = rgb_frame_new[:,:,i] * masks[0]
                
                color_crop = o3d.geometry.Image(rgb_frame_new)
                depth_crop = o3d.geometry.Image(depth_frame_new.astype(np.uint16))
                rgbd_crop = o3d.geometry.RGBDImage.create_from_color_and_depth(color_crop, depth_crop)

                plt.figure()
                plt.subplot(2, 2, 1)
                plt.title('Grayscale scene')
                plt.imshow(rgbd_frame.color)
                plt.subplot(2, 2, 2)
                plt.title('Depth scene')
                plt.imshow(rgbd_frame.depth)
                plt.subplot(2, 2, 3)
                plt.title('Grayscale crop')
                plt.imshow(rgbd_crop.color)
                plt.subplot(2, 2, 4)
                plt.title('Depth crop')
                plt.imshow(rgbd_crop.depth)
                plt.show()

                print("Use Yolact mask to crop point cloud")
                detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, intrinsic)
                # Flip it, otherwise the pointcloud will be upside down
                detected_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                detected_pcd = detected_pcd.voxel_down_sample(self.voxel_size)
                o3d.visualization.draw_geometries([detected_pcd], window_name = 'Yolact PCD')

                if filt_type == 'STATISTICAL':
                    print("Statistical oulier removal")
                    filt_pcd, ind = detected_pcd.remove_statistical_outlier(**filt_params_dict)
                    display_inlier_outlier(detected_pcd, ind)
                    o3d.visualization.draw_geometries([filt_pcd], window_name = 'Filtered PCD')
                elif filt_type == 'RADIUS':
                    print("Radius oulier removal")
                    filt_pcd, ind = detected_pcd.remove_radius_outlier(**filt_params_dict)
                    display_inlier_outlier(detected_pcd, ind)
                    o3d.visualization.draw_geometries([filt_pcd], window_name = 'Filtered PCD')
                else:
                    filt_pcd = copy.deepcopy(detected_pcd)
                    raise Warning('Filtering method (filt_type) not valid')

                

