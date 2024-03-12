import open3d as o3d
import numpy as np


cam1_cl=o3d.io.read_point_cloud("ex2_pointcamera1.xyz")
cam2_cl=o3d.io.read_point_cloud("ex2_pointcamera2.xyz")


plane_model, inliers = cam1_cl.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
inlier_cloud = cam1_cl.select_by_index(inliers)
outlier_cloud = cam1_cl.select_by_index(inliers, invert=True)
inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.002)
outlier_cloud = outlier_cloud.voxel_down_sample(voxel_size=0.002)

o3d.visualization.draw_geometries([inlier_cloud.paint_uniform_color([0, 1., 0]),
                                            outlier_cloud.paint_uniform_color([1., 0, 0])],
                                            window_name = 'Plane (green) - Outliers (red)')
cam1_cl = inlier_cloud


plane_model, inliers = cam2_cl.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
inlier_cloud = cam2_cl.select_by_index(inliers)
outlier_cloud = cam2_cl.select_by_index(inliers, invert=True)
inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size=0.002)
outlier_cloud = outlier_cloud.voxel_down_sample(voxel_size=0.002)

o3d.visualization.draw_geometries([inlier_cloud.paint_uniform_color([0, 1., 0]),
                                            outlier_cloud.paint_uniform_color([1., 0, 0])],
                                            window_name = 'Plane (green) - Outliers (red)')
cam2_cl = inlier_cloud

cam1_cl.paint_uniform_color([0, 1., 0])
cam2_cl.paint_uniform_color([1., 0, 0])
                                            
o3d.visualization.draw_geometries([cam1_cl,cam2_cl],window_name = 'Prima: camera1 (green) - camera2 (red)')
#dist=cam1_cl.compute_point_cloud_distance(cam2_cl)
""" cam1_cl_points = np.asarray(cam1_cl.points)
min_cam1_cl = cam1_cl_points[:,2].min()

     
cam2_cl_points = np.asarray(cam2_cl.points)
min_cam2_cl = cam2_cl_points[:,2].min()
cam2_cl_points[:,2] = cam2_cl_points[:,2] - ((min_cam2_cl - min_cam1_cl)/2)

cam2_cl = o3d.geometry.PointCloud()
cam2_cl.points = o3d.utility.Vector3dVector(cam2_cl_points) """
""" 
dist=cam1_cl.compute_point_cloud_distance(cam2_cl)
cam2_cl_points = np.asarray(cam2_cl.points)
cam2_cl_points[:,2] = cam2_cl_points[:,2] - np.mean(dist)*1.5
cam2_cl = o3d.geometry.PointCloud()
cam2_cl.points = o3d.utility.Vector3dVector(cam2_cl_points) 

o3d.visualization.draw_geometries([cam1_cl,cam2_cl],window_name = 'Dopo: camera1 (green) - camera2 (red)')
print('DISTANCE ', str(np.mean(cam1_cl.compute_point_cloud_distance(cam2_cl)))) """





cam2_cl_down = cam2_cl.voxel_down_sample(voxel_size=0.02)

source=cam2_cl_down
target=cam1_cl




import matplotlib.pyplot as plt
import copy


voxel_size = 0.05
radius_normal = voxel_size * 2
print(":: Estimate normal with search radius %.3f." % radius_normal)
source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))
target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=10))

radius_feature = voxel_size * 5
print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
model_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
obs_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

distance_threshold = voxel_size * 1.5
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


T_gl = glob_rec.transformation


# Apply Global Registration
source_gl = copy.deepcopy(cam2_cl).transform(T_gl)
source_gl.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([source, source_gl, target], window_name = 'Global registration')

trans_init = T_gl
threshold = 2
""" reg_p2p = o3d.pipelines.registration.registration_icp(
		source, target, threshold, trans_init,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = 10000))
print(reg_p2p)
T_icp = reg_p2p.transformation

# Apply Local Registration
source_icp = copy.deepcopy(source).transform(T_icp)
source_icp.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([source, source_icp, target], window_name = 'Local registration Point-To-Point ICP') """


print("Robust point-to-plane ICP")
sigma = 0.1
loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
print("Using robust loss:", loss)
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
conv = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = 10000)
reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
                                                      threshold, trans_init,
                                                      p2l, conv)
print(reg_p2l)
T_icp = reg_p2l.transformation

# Apply Local Registration
source_icp = copy.deepcopy(cam2_cl).transform(T_icp)
source_icp.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([source, source_icp, target], window_name = 'Local registration Point-To-Plane ICP (robust kernel)')

o3d.visualization.draw_geometries([source_icp, target], window_name = 'Transformed')

print('DISTANCE ', str(np.mean(source_icp.compute_point_cloud_distance(target))))
print