
import sparse
from functools import partial
import glob
from multiprocessing import Pool

import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm, tqdm_notebook

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix

from data.transformations import (
    create_voxel_pointcloud,
    normalize_voxel_intensities,
    move_boxes_to_car_space,
    scale_boxes,
    draw_boxes,
    get_semantic_map_around_ego,
)


import warnings
warnings.filterwarnings("ignore")


def generate_data(cfg, level5data, train, val, classes, class_to_stats):
    train_df, train_data_folder = train
    validation_df, validation_data_folder = val

    voxel_size = (cfg.VOXEL_SIZE, cfg.VOXEL_SIZE, cfg.VOXEL_Z_SIZE)
    bev_shape = (cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS)

    print('start loading map')
    map_mask = level5data.map[0]["mask"]
    map_mask.mask()
    print('end loading map')

    def prepare_training_data_for_scene(
            first_sample_token,
            level5data,
            output_folder, bev_shape, voxel_size, z_offset, box_scale,
            classes, class_to_stats):
        """
        Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.


        """
        sample_token = first_sample_token

        while sample_token:
            print('sample_token', sample_token)
            sample = level5data.get("sample", sample_token)

            sample_lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = level5data.get("sample_data", sample_lidar_token)
            lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)

            ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
            calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

            global_from_car = transform_matrix(ego_pose['translation'],
                                               Quaternion(ego_pose['rotation']), inverse=False)

            car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                               Quaternion(calibrated_sensor['rotation']),
                                               inverse=False)

            try:
                lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
                lidar_pointcloud.transform(car_from_sensor)
            except Exception as e:
                print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
                sample_token = sample["next"]
                continue

            bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
            bev = normalize_voxel_intensities(bev)

            boxes = level5data.get_boxes(sample_lidar_token)

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
                       boxes=boxes, classes=classes, z_offset=z_offset, class_to_stats=class_to_stats)

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

    for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
        print("Preparing data into {} using {} workers".format(data_folder, cfg.DATA_GENERATION_WORKERS))
        first_samples = df.first_sample_token.values

        os.makedirs(data_folder, exist_ok=True)

        process_func = partial(
            prepare_training_data_for_scene,
            level5data=level5data,
            output_folder=data_folder,
            bev_shape=bev_shape,
            voxel_size=voxel_size,
            z_offset=cfg.Z_OFFSET,
            box_scale=cfg.BOX_SCALE,
            classes=classes,
            class_to_stats=class_to_stats
        )

        pool = Pool(cfg.DATA_GENERATION_WORKERS)
        for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
            pass
        pool.close()
