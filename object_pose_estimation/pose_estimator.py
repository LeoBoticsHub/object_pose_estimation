import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
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
    """ Object Pose Estimator based on Yolact segmentation and ICP point-cloud registration """
    def __init__(self, camera_type, obj_label, obj_model_path, yolact_weights, voxel_size, flg_plot = False):
        try:
            self.yolact = YolactInference(model_weights=yolact_weights, display_img = flg_plot)
        except:
            raise ValueError('Yolact inizialization error')

        if camera_type == 'REALSENSE':
            self.camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD)
        elif camera_type == 'ZED':
            self.camera = Zed(rgb_resolution=Zed.Resolution.HD)
        else:
            sys.exit("Wrong camera type!")

        self.obj_label = obj_label      # object yolact label
        self.voxel_size = voxel_size    # downsampling voxel size

        print("Load object model")
        try:
            self.model_pcd = o3d.io.read_point_cloud(obj_model_path)
            self.model_pcd = self.model_pcd.translate(-self.model_pcd.get_center())
        except:
            raise ValueError('Error loading object model')
        
        self.model_pcd = self.model_pcd.voxel_down_sample(self.voxel_size) # 1. Points are bucketed into voxels.
                                                                           # 2. Each occupied voxel generates exact one point by averaging all points inside.
       
        print("Camera initialization")
        for i in range(30):
            _, _ = self.camera.get_aligned_frames()


        self.flg_plot = flg_plot

        if self.flg_plot:
            world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
            o3d.visualization.draw_geometries([self.model_pcd, world_frame], window_name = 'Model PCD')
   


    def get_yolact_pcd(self, filt_type, filt_params_dict):
        """ Get object PCD from camera RGBD frames masked by Yolact inference """
        print("Get camera frames")
        rgb_frame, depth_frame = self.camera.get_aligned_frames()
        rgb_frame = np.array(rgb_frame)
        rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth_frame.astype(np.uint16)))
        

        # set intrinsics for open3d
        width = max(depth_frame.shape[0], depth_frame.shape[1])
        height = min(depth_frame.shape[0], depth_frame.shape[1])
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, self.camera.intr['fx'], self.camera.intr['fy'], self.camera.intr['px'], self.camera.intr['py'])

        # save scene pcd
        scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, intrinsic)
            
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
                
                # if self.flg_plot:
                #     plt.figure()
                #     plt.subplot(2, 2, 1)
                #     plt.title('Grayscale scene')
                #     plt.imshow(rgbd_frame.color)
                #     plt.subplot(2, 2, 2)
                #     plt.title('Depth scene')
                #     plt.imshow(rgbd_frame.depth)
                #     plt.subplot(2, 2, 3)
                #     plt.title('Grayscale crop')
                #     plt.imshow(rgbd_crop.color)
                #     plt.subplot(2, 2, 4)
                #     plt.title('Depth crop')
                #     plt.imshow(rgbd_crop.depth)
                #     plt.show()

                print("Use Yolact mask to crop point cloud")
                detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, intrinsic)
                # # Flip it, otherwise the pointcloud will be upside down
                # detected_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                detected_pcd = detected_pcd.voxel_down_sample(self.voxel_size)

                if self.flg_plot:
                    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
                    o3d.visualization.draw_geometries([detected_pcd, world_frame], window_name = 'Yolact PCD')

                

                if filt_type == 'STATISTICAL':
                    print("Statistical oulier removal")
                    filt_pcd, ind = detected_pcd.remove_statistical_outlier(**filt_params_dict)
                    if self.flg_plot:
                        display_inlier_outlier(detected_pcd, ind)
                elif filt_type == 'RADIUS':
                    print("Radius oulier removal")
                    filt_pcd, ind = detected_pcd.remove_radius_outlier(**filt_params_dict)
                    if self.flg_plot:
                        display_inlier_outlier(detected_pcd, ind)
                else:
                    filt_pcd = copy.deepcopy(detected_pcd)
                    print('Filtering method (filt_type) not valid -> No filter applied')

                if self.flg_plot:
                    o3d.visualization.draw_geometries([filt_pcd, world_frame], window_name = 'Filtered PCD')
                
                return filt_pcd, scene_pcd
            else:
                raise Exception("Yolact: Detected more than one object instance.")
        else:
            raise Exception("Yolact: no object detected.")


    def global_registration(self, obs_pcd):
        """ Get a rough alignment between the observed PCD and the model """
        source = self.model_pcd
        target = obs_pcd

        radius_normal = self.voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

        radius_feature = self.voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        obs_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        distance_threshold = self.voxel_size * 1.5
        print(":: RANSAC registration with liberal distance threshold %.3f." % distance_threshold)
        glob_rec = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, model_fpfh, obs_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
            [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1.0))
        print(glob_rec)
        return glob_rec.transformation

    def local_registration(self, obs_pcd, trans_init, max_iteration = 100000, threshold = 0.01, method = 'p2p'):
        """ Refine the alignment between the observed PCD and the model via Point-to-Point ICP """
        print("Apply local registration via ICP")

        source = self.model_pcd
        target = obs_pcd

        if method == 'p2p':
            print("Point-to-point ICP")
            reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = max_iteration))
            print(reg_p2p)
            return reg_p2p.transformation
            
        elif method == 'p2l':
            # print("Point-to-plane ICP")
            # reg_p2l = o3d.pipelines.registration.registration_icp(
            # source, target, threshold, trans_init,
            # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            # o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = max_iteration))

            print("Robust point-to-plane ICP")
            sigma = 0.01
            loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
            print("Using robust loss:", loss)
            p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
            conv = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = max_iteration)
            reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
                                                                  threshold, trans_init,
                                                                  p2l, conv)

            print(reg_p2l)
            return reg_p2l.transformation
        else:
            raise ValueError("Unknown local registration method")
       

    def locate_object(self, filt_type, filt_params_dict, icp_max_iteration = 100000, icp_threshold = 0.2, method = 'p2p'):
        """ Apply the whole pipeline for object pose estimation """
        filt_pcd, scene_pcd = self.get_yolact_pcd(filt_type, filt_params_dict)
        T_gl = self.global_registration(filt_pcd)
        T_icp = self.local_registration(filt_pcd, T_gl, icp_max_iteration, icp_threshold, method)
        return T_icp, filt_pcd, scene_pcd





                

