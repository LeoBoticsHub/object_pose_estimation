import numpy as np
import matplotlib.pyplot as plt
import copy
import open3d as o3d

voxel_size = 0.005

source = o3d.io.read_point_cloud('test/models/drill.ply').voxel_down_sample(voxel_size) 
target = o3d.io.read_point_cloud('test/models/?????.ply').voxel_down_sample(voxel_size) 


o3d.visualization.draw_geometries([source, target])



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
source_gl = copy.deepcopy(source).transform(T_gl)
source_gl.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([source, source_gl, target], window_name = 'Global registration')

trans_init = T_gl
threshold = 0.01
reg_p2p = o3d.pipelines.registration.registration_icp(
		source, target, threshold, trans_init,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = 10000))
print(reg_p2p)
T_icp = reg_p2p.transformation

# Apply Local Registration
source_icp = copy.deepcopy(source).transform(T_icp)
source_icp.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([source, source_icp, target], window_name = 'Local registration Point-To-Point ICP')


print("Robust point-to-plane ICP")
sigma = 0.01
loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
print("Using robust loss:", loss)
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
conv = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness = 10**-6, relative_rmse = 10**-6, max_iteration = 10000)
reg_p2l = o3d.pipelines.registration.registration_icp(source, target,
                                                      threshold, trans_init,
                                                      p2l, conv)
print(reg_p2p)
T_icp = reg_p2p.transformation

# Apply Local Registration
source_icp = copy.deepcopy(source).transform(T_icp)
source_icp.paint_uniform_color([1, 0.706, 0])

o3d.visualization.draw_geometries([source, source_icp, target], window_name = 'Local registration Point-To-Plane ICP (robust kernel)')

