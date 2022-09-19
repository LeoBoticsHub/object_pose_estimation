import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import copy
import pdb
from pathlib import Path
from camera_utils.camera_init import IntelRealsense
from camera_utils.camera_init import Zed
from ai_utils.YolactInference import YolactInference
import open3d as o3d


if __name__ == '__main__':
    camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD)
    yolact_weights = str(Path.home()) + "/Code/Vision/yolact/weights/yolact_plus_resnet50_54_800000.pth"
    yolact_new = YolactInference(model_weights=yolact_weights, score_threshold=0.6, display_img=True)

    while True:
        img = camera.get_rgb()

        yolact_infer = yolact_new.img_inference(img)