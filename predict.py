#
# from apex.fp16_utils import FP16_Optimizer
# from apex import amp
# import sparse
#
# from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import (
#     Conv2dStaticSamePadding,
#     round_filters,
#     round_repeats,
#     drop_connect,
#     get_same_padding_conv2d,
#     get_model_params,
#     efficientnet_params,
#     load_pretrained_weights,
# )
# from datetime import datetime
# from functools import partial
# import glob
# from multiprocessing import Pool
#
# # Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# # even more threads which would lead to a lot of context switching, slowing things down a lot.
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
#
# import matplotlib.pyplot as plt
#
# import pandas as pd
# import cv2
# from PIL import Image
# import numpy as np
# from tqdm import tqdm, tqdm_notebook
# import scipy
# import scipy.ndimage
# import scipy.special
# from scipy.spatial.transform import Rotation as R
#
# from lyft_dataset_sdk.lyftdataset import LyftDataset
# from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
# from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
# from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, recall_precision
#
# from albumentations import (
#     IAAAffine, PadIfNeeded, HorizontalFlip, Rotate, VerticalFlip,
#     RandomSizedCrop, CenterCrop, Crop, Compose, Cutout,
#     Transpose, RandomRotate90, ElasticTransform, GridDistortion, OpticalDistortion,
#     RandomSizedCrop, ShiftScaleRotate, OneOf, CLAHE, RandomContrast,
#     RandomGamma, RandomBrightness, RandomBrightnessContrast, Blur, GaussNoise,
#     ChannelShuffle, InvertImg, MotionBlur, RandomCrop
# )
#
# import warnings
# warnings.filterwarnings("ignore")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# classes = [
#     "car", "motorcycle", "bus", "bicycle", "truck",
#     "pedestrian", "other_vehicle", "animal", "emergency_vehicle"
# ]
#
#
#
#
#
#
#
#
#
# # Some hyperparameters we'll need to define for the system
# voxel_size = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_Z_SIZE)
# z_offset = Z_OFFSET
# bev_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
#
# # We scale down each box so they are more separated when projected into our coarse voxel space.
# box_scale = BOX_SCALE
#
#
#
#
#
#
#
#
#
#
#
#
