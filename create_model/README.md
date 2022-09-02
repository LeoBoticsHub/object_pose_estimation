# Object Dataset Tools

## Introduction

This repository contains pure python scripts to create 3D reconstructed object mesh (.ply) for object sequences filmed with an RGB-D camera. This project can prepare training and testing data for various deep learning projects such as 6D object pose estimation projects singleshotpose, and many object detection (e.g., faster rcnn) and instance segmentation (e.g., mask rcnn) projects. Ideally, if you have realsense cameras and have some experience with MeshLab or Blender, creating your customized dataset should be as easy as executing a few command line arguments.

This codes in this repository implement a raw 3D model acquisition pipeline through aruco markers and ICP registration. The raw 3D model obtained needs to be processed and noise-removed in a mesh processing software.

The codes are currently written for a single object of interest per frame. They can be modified to create a dataset that has several items within a frame.

## Required packages

- numpy
- Cython
- pypng
- scipy
- scikit-learn
- open3d
- scikit-image
- tqdm
- pykdtree
- opencv-python
- opencv-contrib-python
- trimesh

## Create dataset on customized items

### 1. Preparation

**Color** print the pdf with the correctly sized aruco markers (with ID 1-13) in the arucomarkers folder. Affix the markers surrounding the object of interest, as shown in the picture, make sure that you don't have markers with duplicate IDS .


### 2. Record an object sequence

#### Option 1: Record with a realsense camera (SR300 perfered)

The script is provided to record an object video sequence using a compatible realsense camera. Use record2.py for librealsense SDK 2.0:  

```python
python record.py LINEMOD/OBJECTNAME
```
e.g.,

```python
python record.py LINEMOD/sugar
```

to record a sequence of a sugar box. By default, the script records for 40 seconds after a countdown of 5. You can change the recording interval or exit the recording by pressing "q". Please steadily move the camera to get different views of the object while maintaining that 2-3 markers are within the field of view of the camera at any time.

Note that the project assumes all sequences are saved under the folder named "LINEMOD", use other folder names will cause an error to occur.

If you use record.py to create your sequence, color images, depth aligned to color images, and camera parameters will be automatically saved under the directory of the sequence.

#### Option 2: Use an existing sequence or record with other cameras

If you are using other cameras, please put color images (.jpg) in a folder named "JPEGImages" and the **aligned** depth images (uint16 pngs interpolated over a 8m range) in the "depth" folder. Please note that the algorithm assumes the depth images to be  aligned to color images. Name your color images in sequential order from 0.jpg, 1.jpg ... 600.jpg and the corresponding depth images as 0.png ... 600.png, you should also create a file intrinsics.json under the sequence directory and manually input the camera parameters in the format like below:

{"fx": 614.4744262695312, "fy": 614.4745483398438, "height": 480, "width": 640, "ppy": 233.29214477539062, "ppx": 308.8282470703125, "ID": "620201000292"}

If you don't know your camera's intrinsic, you can put a rough estimation in. All parameters required are fx, fy, cx, cy, where commonly fx = fy and equals to the width of the image and cx and cy is the center of the image. For example, for a 640 x 480 resolution image, fx, fy = 640, cx = 320, cy = 240.

An example sequence can be download [HERE](https://drive.google.com/file/d/1BnW4OMR0UlIsaFAjeBuPWrbDgmqV-AY-/view?usp=sharing), create a directory named "LINEMOD", unzip the example sequence, and put the extracted folder (timer) under LINEMOD.

### 3. Obtain frame transforms

Compute transforms for frames at the specified interval (interval can be changed in config/registrationParameters) against the first frame, save the transforms(4*4 homogeneous transforms) as a numpy array (.npy).

```python
python compute_gt_poses.py LINEMOD/sugar
```

### 4. Register all frames and create a mesh for the registered scene.

```python
python register_scene.py LINEMOD/sugar
```
A raw registeredScene.ply will be saved under the specified directory (e.g., LINEMOD/sugar). The registeredScene.ply is a registered pointcloud of the scene that includes the table top, markers, and any other objects exposed during the scanning, with some level of noise removal. The generated pointcloud requires manual processing, as described in step 5.

Alternatively, you can try skipping some manual efforts by trying register_segmented instead of register_scene.

```python
python register_segmented.py LINEMOD/sugar
```
By default, register_segmented attempts to removes all unwanted backgrounds and performs surface reconstruction that converts the registered pointcloud into a triangular mesh. If MESHING is set to false, the script will only attempt to remove background and auto-complete the unseen bottom with a flat surface (If FILLBOTTOM is set to true), and you will need to do step 5.

register_segmented attempts to remove all unwanted backgrounds and auto-complete the unseen bottom with a flat surface (If FILLBOTTOM is set to True). The most important knob to tune is "MAX_RADIUS", which cuts off any depth reading whose Euclidean distance to the center of the aruco markers observed is longer than the value specified. This value is currently set at 0.2 m, if you have a larger object, you may need to increase this value to not cut off parts of your object.
It is possible that manual processing may still be required to refine the pointcloud.


### 5. Process the registered pointcloud manually (Optional)

**(03/03/2019) You can skip step 5 if you are satisfied with the result from running register_segmented.**

The registered pointcloud needs to be processed in MeshLab to
- Remove background that is not of interest,
- Make sure that the processed pointcloud is free of ANY isolated noise.

Once you are satisfied of the resulting pointcloud, translate it to the center of the scene:
- Select 'Filters/Normals, Curvature and Orientations/Transform: Translate, Center, set Origin',
- Select 'Center on scene BBox' under 'Transformation' tab,
- Tick 'Apply to all visible Layers' and then select 'Apply'.

Finally, rotate the pointcloud w.r.t. world reference frame as it is more convenient for your application. You may find useful the two following functionalities:
- 'Filters/Normals, Curvature and Orientations/Transform: Align to principal Axis',
- 'Filters/Normals, Curvature and Orientations/Transform: Rotate'
