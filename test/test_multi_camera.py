import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
from pathlib import Path
from camera_utils.camera_init import IntelRealsense
from camera_utils.camera_init import Zed
from ai_utils.YolactInference import YolactInference
from camera_calibration_lib.cameras_extrinsic_calibration import extrinsic_calibration

try:
    yolact_weights = str(Path.home()) + "/Code/Vision/yolact/weights/yolact_plus_resnet50_drill_74_750.pth"
    yolact = YolactInference(model_weights=yolact_weights, display_img = False)
except:
    raise ValueError('Yolact inizialization error')
obj_label = 'drill'      # object yolact label

camera1 = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD, serial_number='023322061667')
camera2 = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD, serial_number='023322062736')
print("Cameras initialization")
for i in range(30):
    _, _ = camera1.get_aligned_frames()
    _, _ = camera2.get_aligned_frames()

cameras = [camera1, camera2]

try:
    print("Load cameras configuration")
    cam1_H_cam2 = np.loadtxt('test/cam1_H_cam2.csv', delimiter=',')
except:
    print("Loading failed. Cameras re-calibration")
    chess_size = (9, 6)
    chess_square_size = 25
    cam1_H_camX = extrinsic_calibration([camera1, camera2], chess_size, chess_square_size, display_frame = False)
    cam1_H_cam2 = cam1_H_camX[0]
    np.savetxt('test/cam1_H_cam2.csv', cam1_H_cam2, delimiter=',')

print("cam1_H_cam2:\n", cam1_H_cam2)
camera1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
camera2_frame = copy.deepcopy(camera1_frame).transform(cam1_H_cam2)

print("Get camera frames")
pcds = []
camera_count = 1
for camera in cameras:
    rgb_frame, depth_frame = camera.get_aligned_frames()
    rgb_frame = np.array(rgb_frame)
    rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth_frame.astype(np.uint16)))
    # set intrinsics for open3d
    width = max(depth_frame.shape[0], depth_frame.shape[1])
    height = min(depth_frame.shape[0], depth_frame.shape[1])
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, camera.intr['fx'], camera.intr['fy'], camera.intr['px'], camera.intr['py'])

    scene = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, intrinsic)
    o3d.visualization.draw_geometries([scene])

    print("Yolact inference")
    infer = yolact.img_inference(rgb_frame, classes=[obj_label])

    if len(infer) != 0:
        boxes = infer[obj_label]['boxes']
        masks = infer[obj_label]['masks']
        if len(boxes) == 1:
            
            rgb_frame_new = rgb_frame.copy()
            depth_frame_new = depth_frame.copy()
            depth_frame_new = np.array(depth_frame_new * masks[0], dtype = np.uint16)
            
            for i in range(3):
                rgb_frame_new[:,:,i] = rgb_frame_new[:,:,i] * masks[0]
            
            color_crop = o3d.geometry.Image(rgb_frame_new)
            depth_crop = o3d.geometry.Image(depth_frame_new.astype(np.uint16))
            rgbd_crop = o3d.geometry.RGBDImage.create_from_color_and_depth(color_crop, depth_crop, 1000.0, 5.0, False)

            # save scene pcd
            if camera_count == 1:
                pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, intrinsic))
            else:
                pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, intrinsic, extrinsic = np.linalg.inv(cam1_H_cam2)))
                
                # # Passing inv(cam1_H_cam2) as 'extrinsic' is equivalent to apply transform(cam1_H_cam2):
                # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, intrinsic)
                # pcds.append(pcd.transform(cam1_H_cam2))
                
            camera_count += 1

full_pcd = pcds[0] + pcds[1]
voxel_size = 0.01
full_pcd.voxel_down_sample(voxel_size)
full_pcd, ind  = full_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)


o3d.visualization.draw_geometries([full_pcd, camera1_frame, camera2_frame])


o3d.io.write_point_cloud('test/models/comb_drill.ply', full_pcd)