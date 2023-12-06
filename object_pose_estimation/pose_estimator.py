import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import pickle as pkl
from camera_utils.cameras.IntelRealsense import IntelRealsense
# from camera_utils.cameras.Zed import Zed
from ai_utils.detectors.YolactInference import YolactInference
from ai_utils.detectors.Yolov8Inference import Yolov8Inference
from camera_calibration_lib.cameras_extrinsic_calibration import extrinsic_calibration
import open3d as o3d
import cv2


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name = 'Filtering')


def refine_mask(mask):

        # assert a real mask is given not only noise or empty
        assert not(mask is None or mask.shape[0] == 0 or np.sum(mask) < 500)


        #### FROM COMPUTE DIMENSIONS
        # extract rough mask contours
        cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # search for bigger contour (probably the box)
        max_area = 0
        max_cnt = cnt[0]
        for cont in cnt:
            if cv2.contourArea(cont) > max_area:
                max_area = cv2.contourArea(cont)
                max_cnt = cont

        # find the min enclosing rectagle of the mask to have a smoother mask
        rect = cv2.minAreaRect(max_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  # turn into ints

        mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(mask, pts=[box], color=(255, 255, 255))
        rect_mask = np.array(mask, dtype=np.uint8)

        # order the vertices

        # ------------- BOX POINT CLOUD ---------------

        # mask binarization
        rect_mask[rect_mask > 254] = 1
        return rect_mask


class PoseEstimator:
    """ Object Pose Estimator based on Yolact segmentation and ICP point-cloud registration """
    def __init__(self, cameras_dict, obj_label, obj_model_path, yolact_weights, voxel_size,
                 ext_cal_path = 'config/cam1_H_camX.pkl', chess_size = (5, 4), chess_square_size = 40,
                 calib_loops = 100, flg_cal_wait_key = False, flg_plot = False):

        self.cameras = []
        # cameras_dict: dictionary { "serial" : "type" } with type either "REALSENSE" or "ZED"
        for serial, type in cameras_dict.items():
            if type == 'REALSENSE':
                self.cameras.append(IntelRealsense(rgb_resolution = IntelRealsense.Resolution.HD, serial_number = serial))
            elif type == 'ZED':
                self.cameras.append(Zed(rgb_resolution = Zed.Resolution.HD, serial_number = serial))
            else:
                sys.exit(f"{bcolors.FAIL}Wrong camera type!{bcolors.ENDC}")
            
        try:
            self.yolact = YolactInference(model_weights = yolact_weights, classes_white_list = [obj_label], display_img = False)
        except:
            raise ValueError(f"{bcolors.FAIL}Yolact inizialization error{bcolors.ENDC}")

        self.obj_label = obj_label      # object yolact label
        self.voxel_size = voxel_size    # downsampling voxel size

        try:
            self.model_pcd = o3d.io.read_point_cloud(obj_model_path)
            self.model_pcd = self.model_pcd.translate(-self.model_pcd.get_center())
            print(f"{bcolors.OKGREEN}Object model LOADED{bcolors.ENDC}")
        except:
            raise ValueError(f"{bcolors.FAIL}Error loading object model{bcolors.ENDC}")
        
        self.model_pcd = self.model_pcd.voxel_down_sample(self.voxel_size) # 1. Points are bucketed into voxels.
                                                                           # 2. Each occupied voxel generates exact one point by averaging all points inside.
       
        print("Get camera intrinsic parameters")
        self.intrinsic_params = []
        for camera in self.cameras:
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(camera.intr['width'], camera.intr['height'], camera.intr['fx'], camera.intr['fy'], camera.intr['px'], camera.intr['py'])
            self.intrinsic_params.append(intrinsic)
        
        print("Camera initialization")
        for i in range(30):
            for camera in self.cameras:
                _, _ = camera.get_aligned_frames()

        if len(self.cameras) > 1:
            try:
                file = open(ext_cal_path,'rb')
                self.cam1_H_camX = pkl.load(file) # hom. transformation from camera_1 to all other cameras
                file.close()
                print(f"{bcolors.OKGREEN}External calibration configuration LOADED{bcolors.ENDC}")
            except:

                print(f"{bcolors.WARNING}Loading ext. calib. data failed. Cameras re-calibration{bcolors.ENDC}")
                input(f"{bcolors.WARNING}Place the calibration chessboard in the workspace (press ENTER to continue){bcolors.ENDC}")
                self.cam1_H_camX = extrinsic_calibration(self.cameras, chess_size, chess_square_size, loops = calib_loops,
                                                         wait_key = flg_cal_wait_key, display_frame = flg_plot)
                
                filehandler = open(ext_cal_path,"wb")
                pkl.dump(self.cam1_H_camX,filehandler)
                filehandler.close()
                print(f"{bcolors.WARNING}Calibration completed. Exit program.{bcolors.ENDC}")
                sys.exit(0)
        else:
            self.cam1_H_camX = [] # only one camera


        self.flg_plot = flg_plot

        # if self.flg_plot:
        #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
        #     o3d.visualization.draw_geometries([self.model_pcd, world_frame], window_name = 'Model PCD')
   


    def get_yolact_pcd(self, filt_type, filt_params_dict, flg_volume_int = True):
        """ Get object PCD from RGBD frames masked by Yolact inference """
        print("Get frames")
        scene_pcds = []
        obj_pcds = []
        if flg_volume_int:  obj_rgbds = []
        for k in range(len(self.cameras)):
            rgb_frame, depth_frame = self.cameras[k].get_aligned_frames()
            rgb_frame = np.array(rgb_frame)
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth_frame.astype(np.uint16)), convert_rgb_to_intensity=False)
            
            # save scene pcd
            if k == 0: # 1st camera
                scene_pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, self.intrinsic_params[k]))
            else: # other cameras
                scene_pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, self.intrinsic_params[k], extrinsic = np.linalg.inv(self.cam1_H_camX[k-1])))
        
            print("Yolact inference")
            infer = self.yolact.img_inference(rgb_frame)

            if self.flg_plot:
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)

            if len(infer) != 0:
                boxes = infer[self.obj_label]['boxes']
                masks = infer[self.obj_label]['masks']
                if len(boxes) == 1:

                    print("Use Yolact mask to crop point cloud")
                    rgb_frame_new = rgb_frame.copy()
                    depth_frame_new = depth_frame.copy()
                    depth_frame_new = np.array(depth_frame_new * masks[0], dtype = np.uint16)
                    
                    for i in range(3):
                        rgb_frame_new[:,:,i] = rgb_frame_new[:,:,i] * masks[0]
                    
                    color_crop = o3d.geometry.Image(rgb_frame_new)
                    depth_crop = o3d.geometry.Image(depth_frame_new.astype(np.uint16))
                    rgbd_crop = o3d.geometry.RGBDImage.create_from_color_and_depth(color_crop, depth_crop, depth_trunc=1.5, convert_rgb_to_intensity=False)
                    if flg_volume_int:  obj_rgbds.append(rgbd_crop)

                    if k == 0: # 1st camera
                        detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, self.intrinsic_params[k])
                    else: # other cameras
                        detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, self.intrinsic_params[k], extrinsic = np.linalg.inv(self.cam1_H_camX[k-1]))
                    
                    if self.flg_plot:
                        o3d.visualization.draw_geometries([detected_pcd, world_frame], window_name = 'Yolact PCD - Camera '+str(k))
                    
                    
                    # save detected pcd
                    obj_pcds.append(detected_pcd)

                else:
                    raise Exception(f"{bcolors.FAIL}Yolact: Detected more than one object instance{bcolors.ENDC}")
            else:
                raise Exception(f"{bcolors.FAIL}Yolact: no object detected{bcolors.ENDC}")

        if flg_volume_int:
            # volume integration
            obj_volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length = self.voxel_size,
                                                                      sdf_trunc   =  self.voxel_size*5, 
                                                                      color_type  =  o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for k in range(len(obj_rgbds)):
                if k == 0:
                    obj_volume.integrate(obj_rgbds[k], self.intrinsic_params[k], np.eye(4))
                else:
                    obj_volume.integrate(obj_rgbds[k], self.intrinsic_params[k], np.linalg.inv(self.cam1_H_camX[k-1]))
                obj_volume_pcd = obj_volume.extract_point_cloud()
                
        # TODO: do either merge or volume integration, not both!
        # merge PCDs
        whole_obj_pcd = obj_pcds[0]
        whole_scene_pcd = scene_pcds[0]
        for k in range(1,len(obj_pcds)):
            whole_obj_pcd = whole_obj_pcd + obj_pcds[k]
            whole_scene_pcd = whole_scene_pcd + scene_pcds[k]

        whole_obj_pcd = whole_obj_pcd.voxel_down_sample(self.voxel_size)
        whole_scene_pcd = whole_scene_pcd.voxel_down_sample(self.voxel_size)

        if filt_type == 'STATISTICAL':
            print("Statistical oulier removal")
            filt_pcd, ind = whole_obj_pcd.remove_statistical_outlier(**filt_params_dict)
            if flg_volume_int:  obj_volume_filt_pcd, obj_volume_ind = obj_volume_pcd.remove_statistical_outlier(**filt_params_dict)
            if self.flg_plot:
                display_inlier_outlier(whole_obj_pcd, ind)
                if flg_volume_int:  display_inlier_outlier(obj_volume_pcd, obj_volume_ind)
        elif filt_type == 'RADIUS':
            print("Radius oulier removal")
            filt_pcd, ind = whole_obj_pcd.remove_radius_outlier(**filt_params_dict)
            if flg_volume_int:  obj_volume_filt_pcd, obj_volume_ind = obj_volume_pcd.remove_radius_outlier(**filt_params_dict)
            if self.flg_plot:
                display_inlier_outlier(whole_obj_pcd, ind)
                if flg_volume_int:  display_inlier_outlier(obj_volume_pcd, obj_volume_ind)
        else:
            filt_pcd = copy.deepcopy(whole_obj_pcd)
            obj_volume_filt_pcd = copy.deepcopy(obj_volume_pcd)
            print(f"{bcolors.WARNING}Filtering method (filt_type) not valid -> No filter applied{bcolors.ENDC}")

        if self.flg_plot:
            o3d.visualization.draw_geometries([whole_scene_pcd, world_frame], window_name = 'Scene PCD')
            o3d.visualization.draw_geometries([filt_pcd, world_frame], window_name = 'Object PCD (Sum)')
            if flg_volume_int:  o3d.visualization.draw_geometries([obj_volume_filt_pcd, world_frame], window_name = 'Object PCD (Volume integration)')
        
        if flg_volume_int:
            return obj_volume_filt_pcd, whole_scene_pcd
        else:
            return filt_pcd, whole_scene_pcd

        


    def global_registration(self, obs_pcd):
        """ Get a rough alignment between the observed PCD and the model """
        
        # TODO: improve by considering info about a world frame        
        # transl = obs_pcd.get_center()
        # R_flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # T_gl = np.eye(4)
        # T_gl[0:3,3] = transl
        # return np.matmul(T_gl,R_flip)

        source = copy.deepcopy(self.model_pcd)
        target = copy.deepcopy(obs_pcd)

        radius_normal = self.voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

        radius_feature = self.voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        obs_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        distance_threshold = self.voxel_size * 10
        print(":: RANSAC registration with liberal distance threshold %.3f." % distance_threshold)
        glob_rec = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, model_fpfh, obs_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
            [
            # Check if two point clouds build the polygons with similar edge lengths
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.1), # 0 (loose) and 1 (strict)
            # Check if aligned point clouds are closes
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1.0))
        print(glob_rec)

        return glob_rec.transformation

    def local_registration(self, obs_pcd, trans_init, max_iteration = 100000, threshold = 0.01, method = 'p2p'):
        """ Refine the alignment between the observed PCD and the model via Point-to-Point ICP """
        print("Apply local registration via ICP")

        source = copy.deepcopy(self.model_pcd)
        target = copy.deepcopy(obs_pcd)

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



class PCD_Obj_Combined:
    """ Object Pose Estimator based on Yolact segmentation and ICP point-cloud registration """
    def __init__(self, cameras_dict, obj_label, yolact_weights, voxel_size,
                 ext_cal_path = 'config/cam1_H_camX.pkl', chess_size = (5, 4), chess_square_size = 40,
                 calib_loops = 100, flg_cal_wait_key = False, flg_plot = False):

        self.cameras = []
        # cameras_dict: dictionary { "serial" : "type" } with type either "REALSENSE" or "ZED"
        for serial, type in cameras_dict.items():
            if type == 'REALSENSE':
                self.cameras.append(IntelRealsense(camera_resolution = IntelRealsense.Resolution.HD, serial_number = serial))
            elif type == 'ZED':
                self.cameras.append(Zed(rgb_resolution = Zed.Resolution.HD, serial_number = serial))
            else:
                sys.exit(f"{bcolors.FAIL}Wrong camera type!{bcolors.ENDC}")
            
        try:
            self.yolact = YolactInference(model_weights = yolact_weights, score_threshold=0.5, classes_white_list = [obj_label], display_img = True)
        except:
            raise ValueError(f"{bcolors.FAIL}Yolact inizialization error{bcolors.ENDC}")

        self.obj_label = obj_label      # object yolact label
        self.voxel_size = voxel_size    # downsampling voxel size
        
        print("Get camera intrinsic parameters")
        self.intrinsic_params = []
        for camera in self.cameras:
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(camera.intr['width'], camera.intr['height'], camera.intr['fx'], camera.intr['fy'], camera.intr['px'], camera.intr['py'])
            self.intrinsic_params.append(intrinsic)
        
        print("Camera initialization")
        for i in range(30):
            for camera in self.cameras:
                _, _ = camera.get_aligned_frames()

        if len(self.cameras) > 1:
            try:
                file = open(ext_cal_path,'rb')
                self.cam1_H_camX = pkl.load(file) # hom. transformation from camera_1 to all other cameras
                file.close()
                print(f"{bcolors.OKGREEN}External calibration configuration LOADED{bcolors.ENDC}")
            except:

                print(f"{bcolors.WARNING}Loading ext. calib. data failed. Cameras re-calibration{bcolors.ENDC}")
                input(f"{bcolors.WARNING}Place the calibration chessboard in the workspace (press ENTER to continue){bcolors.ENDC}")
                self.cam1_H_camX = extrinsic_calibration(self.cameras, chess_size, chess_square_size, loops = calib_loops,
                                                         wait_key = flg_cal_wait_key, display_frame = flg_plot)
                
                filehandler = open(ext_cal_path,"wb")
                pkl.dump(self.cam1_H_camX,filehandler)
                filehandler.close()
                print(f"{bcolors.WARNING}Calibration completed. Exit program.{bcolors.ENDC}")
                sys.exit(0)
        else:
            self.cam1_H_camX = [] # only one camera


        self.flg_plot = flg_plot

        # if self.flg_plot:
        #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
        #     o3d.visualization.draw_geometries([self.model_pcd, world_frame], window_name = 'Model PCD')
   
    
    
    def get_yolact_pcd(self, filt_type, filt_params_dict, flg_volume_int = True):
        """ Get object PCD from RGBD frames masked by Yolact inference """
        print("Get frames")
        scene_pcds = []
        obj_pcds = []
        if flg_volume_int:  obj_rgbds = []
        for k in range(len(self.cameras)):
            rgb_frame, depth_frame = self.cameras[k].get_aligned_frames()
            rgb_frame = np.array(rgb_frame)
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth_frame.astype(np.uint16)), convert_rgb_to_intensity=False)
            
            # save scene pcd
            if k == 0: # 1st camera
                scene_pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, self.intrinsic_params[k]))
            else: # other cameras
                scene_pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, self.intrinsic_params[k], extrinsic = np.linalg.inv(self.cam1_H_camX[k-1])))
        
            print("Yolact inference")
            infer = self.yolact.img_inference(rgb_frame)

            if self.flg_plot:
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)

            if len(infer) != 0:
                boxes = infer[self.obj_label]['boxes']
                masks = infer[self.obj_label]['masks']
                if len(boxes) == 1:

                    
                    print("Use Yolact mask to crop point cloud")
                    rgb_frame_new = rgb_frame.copy()
                    depth_frame_new = depth_frame.copy()
                    mask = np.asarray(masks[0], np.uint8)
                    ##IF WE WANT A RECTANGULAR MASK
                    #rect_mask = refine_mask(mask)
                    ##
                    # extract mask from rgb
                    for i in range(3):
                        rgb_frame_new[:, :, i] = np.multiply(rgb_frame_new[:, :, i], mask)
                    # extract mask from depth
                    depth_frame_new = np.multiply(depth_frame_new, mask, dtype = np.uint16)
                    
                    """ cv2.namedWindow('Rect Mask', cv2.WINDOW_NORMAL)
                    cv2.imshow('Rect Mask', rect_mask.astype(np.float64))
                    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
                    cv2.imshow('Mask', masks[0])
                    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
                    cv2.imshow('rgb', rgb_frame_new)
                    if cv2.waitKey(1) == ord('q'):
                      print("Closed Image Viewer.")
                      exit(0) """

                    # create open3d point cloud of the box
                    color_crop = o3d.geometry.Image(rgb_frame_new)
                    depth_crop = o3d.geometry.Image(depth_frame_new.astype(np.uint16))
                    rgbd_crop = o3d.geometry.RGBDImage.create_from_color_and_depth(color_crop, depth_crop, depth_trunc=1.5, convert_rgb_to_intensity=False)
                    if flg_volume_int:  obj_rgbds.append(rgbd_crop)

                    if k == 0: # 1st camera
                        detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, self.intrinsic_params[k])
                    else: # other cameras
                        detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, self.intrinsic_params[k], extrinsic = np.linalg.inv(self.cam1_H_camX[k-1]))
                    
                    if self.flg_plot:
                        o3d.visualization.draw_geometries([detected_pcd, world_frame], window_name = 'Yolact PCD - Camera '+str(k))
                    
                    # save detected pcd
                    obj_pcds.append(detected_pcd)

                else:
                    raise Exception(f"{bcolors.FAIL}Yolact: Detected more than one object instance{bcolors.ENDC}")
            else:
                raise Exception(f"{bcolors.FAIL}Yolact: no object detected{bcolors.ENDC}")

        if flg_volume_int:
            # volume integration
            obj_volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length = self.voxel_size,
                                                                      sdf_trunc   =  self.voxel_size*5, 
                                                                      color_type  =  o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for k in range(len(obj_rgbds)):
                if k == 0:
                    obj_volume.integrate(obj_rgbds[k], self.intrinsic_params[k], np.eye(4))
                else:
                    obj_volume.integrate(obj_rgbds[k], self.intrinsic_params[k], np.linalg.inv(self.cam1_H_camX[k-1]))
                obj_volume_pcd = obj_volume.extract_point_cloud()
                
        # TODO: do either merge or volume integration, not both!
        # merge PCDs
        whole_obj_pcd = obj_pcds[0]
        whole_scene_pcd = scene_pcds[0]
        
        if self.flg_plot:
            o3d.visualization.draw_geometries([obj_pcds[0].paint_uniform_color([0, 1., 0])],
                                            window_name = 'One camera')
            o3d.visualization.draw_geometries([obj_pcds[1].paint_uniform_color([1., 0, 0])],
                                            window_name = 'Two camera')

            o3d.visualization.draw_geometries([obj_pcds[0].paint_uniform_color([0, 1., 0]),
                                            obj_pcds[1].paint_uniform_color([1., 0, 0])],
                                            window_name = 'One camera (green) - Two camera (red)')
        for k in range(1,len(obj_pcds)):
            whole_obj_pcd = whole_obj_pcd + obj_pcds[k]
            whole_scene_pcd = whole_scene_pcd + scene_pcds[k]

        ########### VARIATION FOR DOWNSAMPLING
        # whole_obj_pcd_numpy = np.asanyarray(whole_obj_pcd.points)
        # whole_scene_pcd_numpy = np.asanyarray(whole_scene_pcd.points)

        # whole_obj_pcd_downsample = farthest_point_sample(whole_obj_pcd_numpy, 8192)
        # #whole_scene_pcd_downsample = farthest_point_sample(whole_scene_pcd_numpy, 8192)

        # whole_obj_pcd = o3d.geometry.PointCloud()

        # whole_obj_pcd.points = o3d.utility.Vector3dVector(whole_obj_pcd_downsample)
        #whole_scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([whole_scene_pcd_downsample]))

        ###########
        
        whole_obj_pcd = whole_obj_pcd.voxel_down_sample(self.voxel_size)
        whole_scene_pcd = whole_scene_pcd.voxel_down_sample(self.voxel_size)

        if filt_type == 'STATISTICAL':
            print("Statistical oulier removal")
            filt_pcd, ind = whole_obj_pcd.remove_statistical_outlier(**filt_params_dict)
            if flg_volume_int:  obj_volume_filt_pcd, obj_volume_ind = obj_volume_pcd.remove_statistical_outlier(**filt_params_dict)
            if self.flg_plot:
                display_inlier_outlier(whole_obj_pcd, ind)
                if flg_volume_int:  display_inlier_outlier(obj_volume_pcd, obj_volume_ind)
        elif filt_type == 'RADIUS':
            print("Radius oulier removal")
            filt_pcd, ind = whole_obj_pcd.remove_radius_outlier(**filt_params_dict)
            if flg_volume_int:  obj_volume_filt_pcd, obj_volume_ind = obj_volume_pcd.remove_radius_outlier(**filt_params_dict)
            if self.flg_plot:
                display_inlier_outlier(whole_obj_pcd, ind)
                if flg_volume_int:  display_inlier_outlier(obj_volume_pcd, obj_volume_ind)
        else:
            filt_pcd = copy.deepcopy(whole_obj_pcd)
            obj_volume_filt_pcd = copy.deepcopy(obj_volume_pcd)
            print(f"{bcolors.WARNING}Filtering method (filt_type) not valid -> No filter applied{bcolors.ENDC}")

        if self.flg_plot:
            o3d.visualization.draw_geometries([whole_scene_pcd, world_frame], window_name = 'Scene PCD')
            o3d.visualization.draw_geometries([filt_pcd, world_frame], window_name = 'Object PCD (Sum)')
            if flg_volume_int:  o3d.visualization.draw_geometries([obj_volume_filt_pcd, world_frame], window_name = 'Object PCD (Volume integration)')
        
        if flg_volume_int:
            return obj_volume_filt_pcd, whole_scene_pcd
        else:
            return filt_pcd, whole_scene_pcd

    def compute_centroid(self, pcd, display=False):

        # -------------PLANE SEGMENTATION TO REMOVE NOISY OUTLIERS----------------

        try: 
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        except Exception:
            print('\033[91mError during plane generation\033[0m')
            raise AssertionError  # to warn the main that the function has not worked
        if o3d.__version__ == '0.9.0.0':
            inlier_cloud = pcd.select_down_sample(inliers)
            outlier_cloud = pcd.select_down_sample(inliers, invert=True)
        else:
            inlier_cloud = pcd.select_by_index(inliers)
            outlier_cloud = pcd.select_by_index(inliers, invert=True)

        
        ########### VARIATION FOR DOWNSAMPLING
        # inlier_cloud_numpy = np.asanyarray(inlier_cloud.points)
        # outlier_cloud_numpy = np.asanyarray(outlier_cloud.points)

        # inlier_cloud_downsample = farthest_point_sample(inlier_cloud_numpy, 8192)
        # outlier_cloud_downsample = farthest_point_sample(outlier_cloud_numpy, 8192)

        # inlier_cloud = o3d.geometry.PointCloud()
        # outlier_cloud = o3d.geometry.PointCloud()

        # inlier_cloud.points = o3d.utility.Vector3dVector(inlier_cloud_downsample)
        # outlier_cloud.points = o3d.utility.Vector3dVector(outlier_cloud_downsample)

        ###########
        
        inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.002)
        outlier_cloud = outlier_cloud.voxel_down_sample(voxel_size=0.002)

        if display:
            o3d.visualization.draw_geometries([inlier_cloud.paint_uniform_color([0, 1., 0]),
                                            outlier_cloud.paint_uniform_color([1., 0, 0])],
                                            window_name = 'Plane (green) - Outliers (red)')
                                            
        #o3d.io.write_point_cloud('pointin.xyz', inlier_cloud) 
        #o3d.io.write_point_cloud('pointout.xyz', outlier_cloud)                          

        # -------------- CENTROID -------------------
        # box upper surface centroid
        centroid = inlier_cloud.get_center()
        centroid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([centroid]))
        centroid.paint_uniform_color([1.0, 0, 0])
        if display:
            o3d.visualization.draw_geometries([pcd, inlier_cloud,  centroid])


        

class  PCD_Obj_Combined_YOLO:
    """ Object Pose Estimator based on Yolact segmentation and ICP point-cloud registration """
    def __init__(self, cameras_dict, obj_label, yolo_weights, voxel_size,
                 ext_cal_path = 'config/cam1_H_camX.pkl', chess_size = (5, 4), chess_square_size = 40,
                 calib_loops = 100, flg_cal_wait_key = False, flg_plot = False):

        self.cameras = []
        # cameras_dict: dictionary { "serial" : "type" } with type either "REALSENSE" or "ZED"
        for serial, type in cameras_dict.items():
            if type == 'REALSENSE':
                self.cameras.append(IntelRealsense(camera_resolution = IntelRealsense.Resolution.HD, serial_number = serial))
            elif type == 'ZED':
                self.cameras.append(Zed(rgb_resolution = Zed.Resolution.HD, serial_number = serial))
            else:
                sys.exit(f"{bcolors.FAIL}Wrong camera type!{bcolors.ENDC}")
            
        try:
            self.yolo = Yolov8Inference(model_weights = yolo_weights, score_threshold=0.5, classes_white_list = [obj_label], display_img = True)
        except:
            raise ValueError(f"{bcolors.FAIL}Yolact inizialization error{bcolors.ENDC}")

        self.obj_label = obj_label      # object yolact label
        self.voxel_size = voxel_size    # downsampling voxel size
        
        print("Get camera intrinsic parameters")
        self.intrinsic_params = []
        for camera in self.cameras:
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(camera.intr['width'], camera.intr['height'], camera.intr['fx'], camera.intr['fy'], camera.intr['px'], camera.intr['py'])
            self.intrinsic_params.append(intrinsic)
        
        print("Camera initialization")
        for i in range(30):
            for camera in self.cameras:
                _, _ = camera.get_aligned_frames()

        if len(self.cameras) > 1:
            try:
                file = open(ext_cal_path,'rb')
                self.cam1_H_camX = pkl.load(file) # hom. transformation from camera_1 to all other cameras
                file.close()
                print(f"{bcolors.OKGREEN}External calibration configuration LOADED{bcolors.ENDC}")
            except:

                print(f"{bcolors.WARNING}Loading ext. calib. data failed. Cameras re-calibration{bcolors.ENDC}")
                input(f"{bcolors.WARNING}Place the calibration chessboard in the workspace (press ENTER to continue){bcolors.ENDC}")
                self.cam1_H_camX = extrinsic_calibration(self.cameras, chess_size, chess_square_size, loops = calib_loops,
                                                         wait_key = flg_cal_wait_key, display_frame = flg_plot)
                
                filehandler = open(ext_cal_path,"wb")
                pkl.dump(self.cam1_H_camX,filehandler)
                filehandler.close()
                print(f"{bcolors.WARNING}Calibration completed. Exit program.{bcolors.ENDC}")
                sys.exit(0)
        else:
            self.cam1_H_camX = [] # only one camera


        self.flg_plot = flg_plot

        # if self.flg_plot:
        #     world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)
        #     o3d.visualization.draw_geometries([self.model_pcd, world_frame], window_name = 'Model PCD')
   
    
    
    def get_yolo_pcd(self, filt_type, filt_params_dict, flg_volume_int = True):
        """ Get object PCD from RGBD frames masked by Yolo inference """
        print("Get frames")
        scene_pcds = []
        obj_pcds = []
        if flg_volume_int:  obj_rgbds = []
        for k in range(len(self.cameras)):
            rgb_frame, depth_frame = self.cameras[k].get_aligned_frames()
            rgb_frame = np.array(rgb_frame)
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_frame), o3d.geometry.Image(depth_frame.astype(np.uint16)), convert_rgb_to_intensity=False)
            
            # save scene pcd
            if k == 0: # 1st camera
                scene_pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, self.intrinsic_params[k]))
            else: # other cameras
                scene_pcds.append(o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, self.intrinsic_params[k], extrinsic = np.linalg.inv(self.cam1_H_camX[k-1])))
        
            print("Yolo inference")
            infer = self.yolo.img_inference(rgb_frame)

            if self.flg_plot:
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.1)

            if len(infer) != 0:
                boxes = infer[self.obj_label]['boxes']
                masks = infer[self.obj_label]['masks']
                if len(boxes) == 1:

                    
                    print("Use Yolo mask to crop point cloud")
                    rgb_frame_new = rgb_frame.copy()
                    depth_frame_new = depth_frame.copy()
                    mask = np.asarray(masks[0], np.uint8)
                    ##IF WE WANT A RECTANGULAR MASK
                    #rect_mask = refine_mask(mask)
                    ##
                    # extract mask from rgb
                    for i in range(3):
                        rgb_frame_new[:, :, i] = np.multiply(rgb_frame_new[:, :, i], mask)
                    # extract mask from depth
                    depth_frame_new = np.multiply(depth_frame_new, mask, dtype = np.uint16)
                    
                    """ cv2.namedWindow('Rect Mask', cv2.WINDOW_NORMAL)
                    cv2.imshow('Rect Mask', rect_mask.astype(np.float64))
                    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
                    cv2.imshow('Mask', masks[0])
                    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)
                    cv2.imshow('rgb', rgb_frame_new)
                    if cv2.waitKey(1) == ord('q'):
                      print("Closed Image Viewer.")
                      exit(0) """

                    # create open3d point cloud of the box
                    color_crop = o3d.geometry.Image(rgb_frame_new)
                    depth_crop = o3d.geometry.Image(depth_frame_new.astype(np.uint16))
                    rgbd_crop = o3d.geometry.RGBDImage.create_from_color_and_depth(color_crop, depth_crop, depth_trunc=1.5, convert_rgb_to_intensity=False)
                    if flg_volume_int:  obj_rgbds.append(rgbd_crop)

                    if k == 0: # 1st camera
                        detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, self.intrinsic_params[k])
                    else: # other cameras
                        detected_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_crop, self.intrinsic_params[k], extrinsic = np.linalg.inv(self.cam1_H_camX[k-1]))
                    
                    if self.flg_plot:
                        o3d.visualization.draw_geometries([detected_pcd, world_frame], window_name = 'Yolact PCD - Camera '+str(k))
                    
                    # save detected pcd
                    obj_pcds.append(detected_pcd)

                else:
                    raise Exception(f"{bcolors.FAIL}Yolo: Detected more than one object instance{bcolors.ENDC}")
            else:
                raise Exception(f"{bcolors.FAIL}Yolo: no object detected{bcolors.ENDC}")

        if flg_volume_int:
            # volume integration
            obj_volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length = self.voxel_size,
                                                                      sdf_trunc   =  self.voxel_size*5, 
                                                                      color_type  =  o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for k in range(len(obj_rgbds)):
                if k == 0:
                    obj_volume.integrate(obj_rgbds[k], self.intrinsic_params[k], np.eye(4))
                else:
                    obj_volume.integrate(obj_rgbds[k], self.intrinsic_params[k], np.linalg.inv(self.cam1_H_camX[k-1]))
                obj_volume_pcd = obj_volume.extract_point_cloud()
                
        # TODO: do either merge or volume integration, not both!
        # merge PCDs
        whole_obj_pcd = obj_pcds[0]
        whole_scene_pcd = scene_pcds[0]
        
        if self.flg_plot:
            o3d.visualization.draw_geometries([obj_pcds[0].paint_uniform_color([0, 1., 0])],
                                            window_name = 'One camera')
            o3d.visualization.draw_geometries([obj_pcds[1].paint_uniform_color([1., 0, 0])],
                                            window_name = 'Two camera')

            o3d.visualization.draw_geometries([obj_pcds[0].paint_uniform_color([0, 1., 0]),
                                            obj_pcds[1].paint_uniform_color([1., 0, 0])],
                                            window_name = 'One camera (green) - Two camera (red)')
        for k in range(1,len(obj_pcds)):
            whole_obj_pcd = whole_obj_pcd + obj_pcds[k]
            whole_scene_pcd = whole_scene_pcd + scene_pcds[k]

        ########### VARIATION FOR DOWNSAMPLING
        # whole_obj_pcd_numpy = np.asanyarray(whole_obj_pcd.points)
        # whole_scene_pcd_numpy = np.asanyarray(whole_scene_pcd.points)

        # whole_obj_pcd_downsample = farthest_point_sample(whole_obj_pcd_numpy, 8192)
        # #whole_scene_pcd_downsample = farthest_point_sample(whole_scene_pcd_numpy, 8192)

        # whole_obj_pcd = o3d.geometry.PointCloud()

        # whole_obj_pcd.points = o3d.utility.Vector3dVector(whole_obj_pcd_downsample)
        #whole_scene_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([whole_scene_pcd_downsample]))

        ###########
        
        whole_obj_pcd = whole_obj_pcd.voxel_down_sample(self.voxel_size)
        whole_scene_pcd = whole_scene_pcd.voxel_down_sample(self.voxel_size)

        if filt_type == 'STATISTICAL':
            print("Statistical oulier removal")
            filt_pcd, ind = whole_obj_pcd.remove_statistical_outlier(**filt_params_dict)
            if flg_volume_int:  obj_volume_filt_pcd, obj_volume_ind = obj_volume_pcd.remove_statistical_outlier(**filt_params_dict)
            if self.flg_plot:
                display_inlier_outlier(whole_obj_pcd, ind)
                if flg_volume_int:  display_inlier_outlier(obj_volume_pcd, obj_volume_ind)
        elif filt_type == 'RADIUS':
            print("Radius oulier removal")
            filt_pcd, ind = whole_obj_pcd.remove_radius_outlier(**filt_params_dict)
            if flg_volume_int:  obj_volume_filt_pcd, obj_volume_ind = obj_volume_pcd.remove_radius_outlier(**filt_params_dict)
            if self.flg_plot:
                display_inlier_outlier(whole_obj_pcd, ind)
                if flg_volume_int:  display_inlier_outlier(obj_volume_pcd, obj_volume_ind)
        else:
            filt_pcd = copy.deepcopy(whole_obj_pcd)
            obj_volume_filt_pcd = copy.deepcopy(obj_volume_pcd)
            print(f"{bcolors.WARNING}Filtering method (filt_type) not valid -> No filter applied{bcolors.ENDC}")

        if self.flg_plot:
            o3d.visualization.draw_geometries([whole_scene_pcd, world_frame], window_name = 'Scene PCD')
            o3d.visualization.draw_geometries([filt_pcd, world_frame], window_name = 'Object PCD (Sum)')
            if flg_volume_int:  o3d.visualization.draw_geometries([obj_volume_filt_pcd, world_frame], window_name = 'Object PCD (Volume integration)')
        
        if flg_volume_int:
            return obj_volume_filt_pcd, whole_scene_pcd
        else:
            return filt_pcd, whole_scene_pcd

    def compute_centroid(self, pcd, display=False):

        # -------------PLANE SEGMENTATION TO REMOVE NOISY OUTLIERS----------------

        try: 
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        except Exception:
            print('\033[91mError during plane generation\033[0m')
            raise AssertionError  # to warn the main that the function has not worked
        if o3d.__version__ == '0.9.0.0':
            inlier_cloud = pcd.select_down_sample(inliers)
            outlier_cloud = pcd.select_down_sample(inliers, invert=True)
        else:
            inlier_cloud = pcd.select_by_index(inliers)
            outlier_cloud = pcd.select_by_index(inliers, invert=True)

        
        ########### VARIATION FOR DOWNSAMPLING
        # inlier_cloud_numpy = np.asanyarray(inlier_cloud.points)
        # outlier_cloud_numpy = np.asanyarray(outlier_cloud.points)

        # inlier_cloud_downsample = farthest_point_sample(inlier_cloud_numpy, 8192)
        # outlier_cloud_downsample = farthest_point_sample(outlier_cloud_numpy, 8192)

        # inlier_cloud = o3d.geometry.PointCloud()
        # outlier_cloud = o3d.geometry.PointCloud()

        # inlier_cloud.points = o3d.utility.Vector3dVector(inlier_cloud_downsample)
        # outlier_cloud.points = o3d.utility.Vector3dVector(outlier_cloud_downsample)

        ###########
        
        inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.002)
        outlier_cloud = outlier_cloud.voxel_down_sample(voxel_size=0.002)

        if display:
            o3d.visualization.draw_geometries([inlier_cloud.paint_uniform_color([0, 1., 0]),
                                            outlier_cloud.paint_uniform_color([1., 0, 0])],
                                            window_name = 'Plane (green) - Outliers (red)')
                                            
        #o3d.io.write_point_cloud('pointin.xyz', inlier_cloud) 
        #o3d.io.write_point_cloud('pointout.xyz', outlier_cloud)                          

        # -------------- CENTROID -------------------
        # box upper surface centroid
        centroid = inlier_cloud.get_center()
        centroid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([centroid]))
        centroid.paint_uniform_color([1.0, 0, 0])
        if display:
            o3d.visualization.draw_geometries([pcd, inlier_cloud,  centroid])        


    

                

