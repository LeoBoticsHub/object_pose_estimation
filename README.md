# object_pose_estimation

Library to perform 6D object pose estimation: Yolact masking + ICP Point Cloud Registration
___

### Dependencies
___
The library contains python modules which are dependent on the following 3rd-party libraries:
```
setuptools, numpy, open3d, opencv-python, matplotlib
```
Additionally, it is necessary to install our [camera_utils](https://github.com/IASRobolab/camera_utils), [ai_utils](https://github.com/IASRobolab/ai_utils), and [camera_calibration](https://github.com/IASRobolab/camera_calibration)

### Installation
___
To install the camera_calibration package on your system, clone the GitHub repository in a folder of your choice, open the cloned repository path in a terminal and run the following command

```
python3 -m pip install .
```

Instead if you want to install the package in "editable" or "develop" mode (to prevent the uninstall/install of the
package at every pkg modification) you have can run the following command:

```
python3 -m pip install -e .
```

## Usage
___

A __PoseEstimator__ object provides different functions to capture object point clouds from RGBD cameras and retrieve object 6D pose using ICP registration techniques (given a model of the desired object).

- __get_yolact_pcd__ use Yolact algorithm to isolate the desired object point cloud from images taken by multiple cameras.
- __global_registration__ gets a rough alignment between the observed point cloud and the model of the desired object.
- __local_registration__ refines the alignment between the observed point cloud and the model using ICP (Iterative Closest Point)
- __locate_object__ applies the full pipeline for object pose estimation (__get_yolact_pcd__ + __global_registration__ + __local_registration__)

Some example scripts can be found inside the __example__ folder.

## License
___
Distributed under the ```GPLv3``` License. See [LICENSE](LICENSE) for more information.

## Authors
___
The package is provided by:

- [Fabio Amadio](https://github.com/fabio-amadio) [Mantainer]
