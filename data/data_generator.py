#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Edit these to point at where the Level 5 dataset lives on your machine.
# DATASET_VERSION = 'v1.02-train'
# DATASET_ROOT = '../level5dataset/v1.02-train/'

# DATASET_ROOT = '/media/ml_data/projects/lyft'
DATASET_ROOT = '/root/data/lyft'


IMG_SIZE = 1024
VOXEL_SIZE = 0.2
VOXEL_Z_SIZE = 0.7
IMG_CHANNELS = 6

IMG_SIZE_CROP = 512

EPOCHS = 128
BATCH_SIZE = 16

REG_COEF = 0.05
BOX_SCALE = 0.9
Z_OFFSET = -2.0

NUM_WORKERS = 16

ADJACENT = True

if ADJACENT:
    ARTIFACTS_FOLDER = "../artifacts_IMG_SIZE_{}_VOXEL_{:.2f}_VOXEL_Z_{:.2f}_HEIGHT_{:.2f}_BOX_SCALE_{:.2f}_ADJ_as_channels".format(
        IMG_SIZE, VOXEL_SIZE, VOXEL_Z_SIZE, VOXEL_Z_SIZE * IMG_CHANNELS, BOX_SCALE)
else:
    ARTIFACTS_FOLDER = "../artifacts_IMG_SIZE_{}_VOXEL_{:.2f}_VOXEL_Z_{:.2f}_HEIGHT_{:.2f}_BOX_SCALE_{:.2f}".format(
        IMG_SIZE, VOXEL_SIZE, VOXEL_Z_SIZE, VOXEL_Z_SIZE * IMG_CHANNELS, BOX_SCALE)


classes = [
        "car", "motorcycle", "bus", "bicycle", "truck",
        "pedestrian", "other_vehicle", "animal", "emergency_vehicle"
    ]


while_list_classes = [
    "car",
    "bus",
    "truck",
    "other_vehicle",
    "emergency_vehicle",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "animal",
]

print('while_list_classes', while_list_classes)

class_to_size = {
    "car":               (1.93,  4.76, 1.72),
    "motorcycle":        (0.96,  2.35, 1.59),
    "bus":               (2.96, 12.34, 3.44),
    "bicycle":           (0.63,  1.76, 1.44),
    "truck":             (2.84, 10.24, 3.44),
    "pedestrian":        (0.77,  0.81, 1.78),
    "other_vehicle":     (2.79,  8.20, 3.23),
    "animal":            (0.36,  0.73, 0.51),
    "emergency_vehicle": (2.45,  6.52, 2.39)
}

class_to_width = {class_name: class_to_size[class_name][0] for class_name in class_to_size}
class_to_len = {class_name: class_to_size[class_name][1] for class_name in class_to_size}
class_to_height = {class_name: class_to_size[class_name][2] for class_name in class_to_size}

print('classes', classes)

from apex.fp16_utils import FP16_Optimizer
from apex import amp
import sparse

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (
    Conv2dStaticSamePadding,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)
from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, recall_precision


import warnings
warnings.filterwarnings("ignore")


# ## A. Creating an index and splitting into train and validation scenes

# In[3]:


level5data = LyftDataset(json_path=DATASET_ROOT + "/train_data/", data_path=DATASET_ROOT, verbose=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)


# We create a Pandas DataFrame with a row for each of the scenes

# In[4]:


records = [(level5data.get('sample', record['first_sample_token'])['timestamp'], record) for record in
        level5data.scene]

entries = []

for start_time, record in sorted(records):
    start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000

    token = record['token']
    name = record['name']
    date = datetime.utcfromtimestamp(start_time)
    host = "-".join(record['name'].split("-")[:2])
    first_sample_token = record["first_sample_token"]

    entries.append((host, name, date, token, first_sample_token))
            
df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])


# In[5]:


host_count_df = df.groupby("host")['scene_token'].count()
print(host_count_df)


# ### Train/Validation split
# Let's split the data by car to get a validation set.
# Alternatively we could consider doing it by scenes, date, or completely randomly.

# In[6]:


validation_hosts = ["host-a007", "host-a008", "host-a009"]

validation_df = df[df["host"].isin(validation_hosts)]
vi = validation_df.index
train_df = df[~df.index.isin(vi)]


sample_token = train_df.first_sample_token.values[0]
sample = level5data.get("sample", sample_token)

sample_lidar_token = sample["data"]["LIDAR_TOP"]
lidar_data = level5data.get("sample_data", sample_lidar_token)
lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

# Homogeneous transformation matrix from car frame to world frame.
global_from_car = transform_matrix(ego_pose['translation'],
                                   Quaternion(ego_pose['rotation']), inverse=False)

# Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                    inverse=False)

lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

# The lidar pointcloud is defined in the sensor's reference frame.
# We want it in the car's reference frame, so we transform each point
lidar_pointcloud.transform(car_from_sensor)

# A sanity check, the points should be centered around 0 in car space.
plt.hist(lidar_pointcloud.points[0], alpha=0.5, bins=30, label="X")
plt.hist(lidar_pointcloud.points[1], alpha=0.5, bins=30, label="Y")
plt.legend()
plt.xlabel("Distance from car along axis")
plt.ylabel("Amount of points")
plt.show()

map_mask = level5data.map[0]["mask"]


def get_semantic_map_around_ego(map_mask, ego_pose, voxel_size, output_shape):

    def crop_image(image: np.array,
                           x_px: int,
                           y_px: int,
                           axes_limit_px: int) -> np.array:
                x_min = int(x_px - axes_limit_px)
                x_max = int(x_px + axes_limit_px)
                y_min = int(y_px - axes_limit_px)
                y_max = int(y_px + axes_limit_px)

                cropped_image = image[y_min:y_max, x_min:x_max]

                return cropped_image

    pixel_coords = map_mask.to_pixel_coords(ego_pose['translation'][0], ego_pose['translation'][1])

    extent = voxel_size*output_shape[0]*0.5
    scaled_limit_px = int(extent * (1.0 / (map_mask.resolution)))
    mask_raster = map_mask.mask()

    cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * np.sqrt(2)))

    ypr_rad = Quaternion(ego_pose['rotation']).yaw_pitch_roll
    yaw_deg = -np.degrees(ypr_rad[0])

    rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
    ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2,
                                 scaled_limit_px)[::-1]
    
    ego_centric_map = cv2.resize(ego_centric_map, output_shape[:2], cv2.INTER_NEAREST)
    return ego_centric_map.astype(np.float32)/255
    
ego_centric_map = get_semantic_map_around_ego(
    map_mask, ego_pose, voxel_size=VOXEL_SIZE, output_shape=(IMG_SIZE, IMG_SIZE)
) 

plt.imshow(ego_centric_map)
plt.show()


# As input for our network we voxelize the LIDAR points. That means that we go from a list of coordinates of points, to a X by Y by Z space.

# In[13]:


def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
    """
    Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
    Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.
    
    An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
    """
    
    shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)
    
    tm = np.eye(4, dtype=np.float32)
    translation = shape/2 + offset/voxel_size
    
    tm = tm * np.array(np.hstack((1/voxel_size, [1])))

    tm[:3, 3] = np.transpose(translation)
    return tm


def transform_points(points, transf_matrix):
    """
    Transform (3,N) or (4,N) points using transformation matrix.
    """
    if points.shape[0] not in [3,4]:
        raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
    return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]

# Let's try it with some example values
tm = create_transformation_matrix_to_voxel_space(shape=(100,100,4), voxel_size=(0.5,0.5,0.5), offset=(0,0,0.5))
p = transform_points(np.array([[10, 10, 0, 0, 0], [10, 5, 0, 0, 0],[0, 0, 0, 2, 0]], dtype=np.float32), tm)
print(p)


# In[14]:


def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
    if len(shape) != 3:
        raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")
        
    if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
        raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

    tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
    p = transform_points(points, tm)
    return p


def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):

    points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
    points_voxel_coords = points_voxel_coords[:3].transpose(1,0)
    points_voxel_coords = np.int0(points_voxel_coords)
    
    bev = np.zeros(shape, dtype=np.float32)
    bev_shape = np.array(shape)

    within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))
    
    points_voxel_coords = points_voxel_coords[within_bounds]
    coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)
        
    # Note X and Y are flipped:
    bev[coord[:,1], coord[:,0], coord[:,2]] = count
    
    return bev

def normalize_voxel_intensities(bev, max_intensity=16):
    return (bev/max_intensity).clip(0,1)


voxel_size = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_Z_SIZE)
z_offset = Z_OFFSET
bev_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)

# So that the values in the voxels range from 0,1 we set a maximum intensity.
bev = normalize_voxel_intensities(bev)


boxes = level5data.get_boxes(sample_lidar_token)

target_im = np.zeros((*bev.shape[:2], 3), dtype=np.uint8)


def move_boxes_to_car_space(boxes, ego_pose):
    """
    Move boxes from world space to car space.
    Note: mutates input boxes.
    """
    translation = -np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation']).inverse
    
    for box in boxes:
        # Bring box to car space
        box.translate(translation)
        # box.rotate(rotation)
        box.rotate_around_origin(rotation)


def scale_boxes(boxes, factor):
    """
    Note: mutates input boxes
    """
    for box in boxes:
        box.wlh[:2] = box.wlh[:2] * factor


def draw_boxes(im, wlh, voxel_size, boxes, classes, z_offset=0.0):
    im_w, im_l, im_h, im_w_ave, im_l_ave, im_h_ave = wlh
    
    for box in boxes:
        corners = box.bottom_corners()
        corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1, 0)
        corners_voxel = corners_voxel[:,:2] # Drop z coord
        
        class_color = classes.index(box.name) + 1
        
        if class_color == 0:
            raise Exception("Unknown class: {}".format(box.name))

        cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)

        w, l, h = box.wlh
        
        w_ave, l_ave, h_ave = class_to_width[box.name], class_to_len[box.name], class_to_height[box.name]
        
        cv2.drawContours(im_w, np.int0([corners_voxel]), 0, (w), -1)
        cv2.drawContours(im_l, np.int0([corners_voxel]), 0, (l), -1)
        cv2.drawContours(im_h, np.int0([corners_voxel]), 0, (h), -1)

        cv2.drawContours(im_w_ave, np.int0([corners_voxel]), 0, (w_ave), -1)
        cv2.drawContours(im_l_ave, np.int0([corners_voxel]), 0, (l_ave), -1)
        cv2.drawContours(im_h_ave, np.int0([corners_voxel]), 0, (h_ave), -1)

        
target_w = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
target_l = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
target_h = np.zeros((*bev.shape[:2], 1), dtype=np.float32)

target_w_ave = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
target_l_ave = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
target_h_ave = np.zeros((*bev.shape[:2], 1), dtype=np.float32)

move_boxes_to_car_space(boxes, ego_pose)
scale_boxes(boxes, 0.8)

target_wlh = (
    target_w, target_l, target_h, 
    target_w_ave, target_l_ave, target_h_ave
)

draw_boxes(target_im, target_wlh,
           voxel_size, boxes, classes, z_offset=z_offset)

target_wlh = np.concatenate((target_w, target_l, target_h), axis=2)
target_wlh_ave = np.concatenate((target_w_ave, target_l_ave, target_h_ave), axis=2)


# In[18]:


np.sum((target_wlh[:, :, 2] - target_wlh_ave[:, :, 2])[target_im[:, :, 2] > 0] > 0)


# In[19]:


np.sum((target_wlh[:, :, 2] - target_wlh_ave[:, :, 2])[target_im[:, :, 2] > 0] < 0)


# In[20]:


plt.figure(figsize=(8, 8))
plt.imshow((target_im > 0).astype(np.float32), cmap='Set2')
plt.show()


# In[21]:


plt.figure(figsize=(8,8))
plt.imshow((target_h.squeeze()).astype(np.float32), cmap='Set2')
plt.show()


# In[ ]:





# These are the annotations in the same top-down frame, Below we plot the same scene using the NuScenes SDK. Don't worry about it being flipped.

# In[22]:


def visualize_lidar_of_sample(sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)
visualize_lidar_of_sample(sample_token)


# Now we will run this on all samples in the train and validation set, and write the input and target images to their respective folders.

# In[23]:


# Some hyperparameters we'll need to define for the system
voxel_size = (VOXEL_SIZE, VOXEL_SIZE, VOXEL_Z_SIZE)
z_offset = Z_OFFSET
bev_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

# We scale down each box so they are more separated when projected into our coarse voxel space.
box_scale = BOX_SCALE


# In[24]:


# "bev" stands for birds eye view
train_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_train_data")
validation_data_folder = os.path.join(ARTIFACTS_FOLDER, "./bev_validation_data")


# In[27]:



def get_global_point_cloud(sample_token):
    local_sample = level5data.get("sample", sample_token)

    local_sample_lidar_token = local_sample["data"]["LIDAR_TOP"]
    local_lidar_data = level5data.get("sample_data", local_sample_lidar_token)
    local_lidar_filepath = level5data.get_sample_data_path(local_sample_lidar_token)

    local_ego_pose = level5data.get("ego_pose", local_lidar_data["ego_pose_token"])
    local_calibrated_sensor = level5data.get("calibrated_sensor", local_lidar_data["calibrated_sensor_token"])

    global_from_car = transform_matrix(local_ego_pose['translation'],
                                       Quaternion(local_ego_pose['rotation']),
                                       inverse=False)

    car_from_sensor = transform_matrix(local_calibrated_sensor['translation'],
                                       Quaternion(local_calibrated_sensor['rotation']),
                                       inverse=False)

    local_lidar_pointcloud = LidarPointCloud.from_file(local_lidar_filepath)

    local_lidar_pointcloud.transform(car_from_sensor)
    local_lidar_pointcloud.transform(global_from_car)

    return local_lidar_pointcloud


def prepare_training_data_for_scene(first_sample_token, output_folder, bev_shape, voxel_size, z_offset, box_scale):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.
    

    """
    sample_token = first_sample_token
    
    while sample_token:
        # print(sample_token)
        sample = level5data.get("sample", sample_token)

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])

        car_from_global = transform_matrix(
            ego_pose['translation'],
            Quaternion(ego_pose['rotation']), inverse=True
        )

        try:
            lidar_pointcloud = get_global_point_cloud(sample_token)
            lidar_pointcloud.transform(car_from_global)
            points = np.array(lidar_pointcloud.points)

            if ADJACENT:
                if sample['prev']:
                    lidar_pointcloud = get_global_point_cloud(sample['prev'])
                    lidar_pointcloud.transform(car_from_global)
                    adj_points = np.array(lidar_pointcloud.points)

                    # prev_sample = level5data.get("sample", sample['prev'])
                    # if prev_sample['prev']:
                    #     lidar_pointcloud = get_global_point_cloud(prev_sample['prev'])
                    #     lidar_pointcloud.transform(car_from_global)
                    #     prev_points = np.array(lidar_pointcloud.points)
                    #     np.hstack((adj_points, prev_points))

                else:
                    sample_token = sample["next"]
                    continue

                if sample['next']:
                    lidar_pointcloud = get_global_point_cloud(sample['next'])
                    lidar_pointcloud.transform(car_from_global)
                    next_points = np.array(lidar_pointcloud.points)
                    adj_points = np.hstack((adj_points, next_points))

                    # next_sample = level5data.get("sample", sample['next'])
                    # if next_sample['next']:
                    #     lidar_pointcloud = get_global_point_cloud(next_sample['next'])
                    #     lidar_pointcloud.transform(car_from_global)
                    #     next_points = np.array(lidar_pointcloud.points)
                    #     adj_points = np.hstack((adj_points, next_points))
                else:
                    sample_token = sample["next"]
                    continue

        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue

        bev = create_voxel_pointcloud(points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        if ADJACENT:
            adj_bev = create_voxel_pointcloud(adj_points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
            adj_bev = normalize_voxel_intensities(adj_bev)
            bev = np.concatenate((bev, adj_bev), axis=2)

        # print(bev.shape)

        boxes = level5data.get_boxes(sample_lidar_token)

        boxes = list(filter(lambda x: x.name in while_list_classes, boxes))

        if len(boxes) == 0:
            sample_token = sample["next"]
            continue

        target = np.zeros((*bev.shape[:2], 3))

        target_w = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
        target_l = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
        target_h = np.zeros((*bev.shape[:2], 1), dtype=np.float32)

        target_w_ave = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
        target_l_ave = np.zeros((*bev.shape[:2], 1), dtype=np.float32)
        target_h_ave = np.zeros((*bev.shape[:2], 1), dtype=np.float32)

        target_wlh = target_w, target_l, target_h, target_w_ave, target_l_ave, target_h_ave
        
        move_boxes_to_car_space(boxes, ego_pose)
        
        scale_boxes(boxes, box_scale)
        
        draw_boxes(target, target_wlh, voxel_size, 
                   boxes=boxes, classes=classes, z_offset=z_offset)

        target_wlh = np.concatenate(
            (target_w, target_l, target_h, target_w_ave, target_l_ave, target_h_ave), 
            axis=2
        ).astype(np.float16)
        
        bev_im = np.round(bev * 255).astype(np.uint8)
        target_im = target[:, :, 0]  # take one channel only

        assert np.sum((target_wlh)[target_im > 0] == 0) == 0

        semantic_im = get_semantic_map_around_ego(map_mask, ego_pose, voxel_size[0], target_im.shape)
        semantic_im = np.round(semantic_im * 255).astype(np.uint8)

        bev_im_sparse = sparse.COO(bev_im)
        sparse.save_npz(os.path.join(output_folder, "{}_input.npz".format(sample_token)), bev_im_sparse)

        target_wlh_sparse = sparse.COO(target_wlh)
        sparse.save_npz(os.path.join(output_folder, "{}_target_wlh.npz".format(sample_token)), target_wlh_sparse)

        cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)
        cv2.imwrite(os.path.join(output_folder, "{}_map.png".format(sample_token)), semantic_im)
        
        sample_token = sample["next"]

        
if True: 
    for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
        print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
        first_samples = df.first_sample_token.values

        os.makedirs(data_folder, exist_ok=True)

        process_func = partial(prepare_training_data_for_scene,
                               output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale)

        pool = Pool(NUM_WORKERS)
        for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
            pass
        pool.close()




