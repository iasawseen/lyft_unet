import os
import json

from data.dataset import get_dataloader
from configs.config import get_cfg

import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from models.utils import get_unet_model, DataParallelCustom

from scipy.spatial.transform import Rotation as R

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D, recall_precision
from data.transformations import create_transformation_matrix_to_voxel_space, transform_points
from multiprocessing import Pool
from functools import partial
from utils.mAP_evaluation import get_average_precisions
from data.stats import get_classes, get_class_stats
from utils.tta import TTAWrapper, d4_image2mask, fliplr_image2mask

from multiprocessing import Process

from data.dataset import get_dataloaders

import warnings
warnings.filterwarnings("ignore")


def predict(fold=0):
    print('Predicting fold={}\n\n\n'.format(fold))

    cfg = get_cfg()

    classes = get_classes()
    class_to_width, class_to_len, class_to_height = get_class_stats()

    level5data = LyftDataset(
        json_path=cfg.DATASET_ROOT + "/train_data/",
        data_path=cfg.DATASET_ROOT,
        verbose=False
    )

    _, validation_dataloader = get_dataloaders(cfg, fold=fold, val_ratio=0.01)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_unet_model(
        cfg,
        cfg.IMG_CHANNELS * 2 + 3,
        num_output_classes=10,
        backbone_name=cfg.BACKBONE,
        dropout=cfg.DROPOUT,
        input_dropout=cfg.INPUT_DROPOUT,
    )

    model = model.to(device)

    checkpoint_filename = '/media/ml_data/projects/lyft_unet/logs/' \
        'all_classes_v0.3_drop_0.33_fold_{fold}/checkpoints/best.pth'.format(fold=fold)

    checkpoint_filepath = os.path.join(
        cfg.ARTIFACTS_FOLDER,
        checkpoint_filename
    )

    model.load_state_dict(torch.load(checkpoint_filepath)['model_state_dict'])
    model = DataParallelCustom(model)
    model = TTAWrapper(model, fliplr_image2mask)

    progress_bar = tqdm(validation_dataloader)

    predictions = np.zeros(
        (len(validation_dataloader) * validation_dataloader.batch_size, 1 + len(classes), cfg.IMG_SIZE, cfg.IMG_SIZE),
        dtype=np.uint8
    )

    if cfg.REGRESSION:
        predictions_reg = np.zeros(
            (len(validation_dataloader) * validation_dataloader.batch_size, 1, cfg.IMG_SIZE, cfg.IMG_SIZE),
            dtype=np.uint8
        )

    sample_tokens = []

    with torch.no_grad():
        model.eval()
        for ii, batch in enumerate(progress_bar):
            X = batch['image']
            batch_sample_tokens = batch['token']

            batch_size = X.size(0)
            offset = ii * batch_size
            sample_tokens.extend(batch_sample_tokens)

            X = X.to(device)  # [N, 1, H, W]

            prediction = model(X)  # [N, 2, H, W]

            if cfg.REGRESSION:
                prediction, prediction_reg = prediction['logits'], prediction['logits_reg']
            else:
                prediction = prediction['logits']

            prediction = F.softmax(prediction, dim=1)

            prediction = prediction.cpu().numpy()

            if cfg.REGRESSION:
                prediction_reg = prediction_reg.cpu().numpy()

            prediction = np.round(prediction * 255).astype(np.uint8)
            predictions[offset: offset + batch_size] = prediction

            if cfg.REGRESSION:
                predictions_reg[offset: offset + batch_size] = prediction_reg

    predictions.dump('preds_fold_{}.npy'.format(fold))

    detection_boxes = []
    detection_scores = []
    detection_classes = []

    for i in tqdm(range(len(predictions))):
        prediction = predictions[i]

        threshold = 0.5
        background_threshold = int(255 * threshold)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        prediction_non_class0 = 255 - prediction[0]
        thresholded_p = (prediction_non_class0 > background_threshold).astype(np.uint8)
        prediction_opened = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel)

        sample_boxes = []
        sample_detection_scores = []
        sample_detection_classes = []

        contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)

            box_center_index = np.int0(np.mean(box, axis=0))

            for class_index in range(len(classes)):
                mask = np.zeros(prediction.shape[1:]).astype(np.uint8)
                cv2.drawContours(mask, np.int0([box]), 0, 255, -1)

                box_detection_score = prediction[class_index + 1, box_center_index[1], box_center_index[0]] / 255

                if box_detection_score < 0.01:
                    continue

                box_center_class = classes[class_index]

                sample_detection_classes.append(box_center_class)
                sample_detection_scores.append(box_detection_score)
                sample_boxes.append(box)

        detection_boxes.append(np.array(sample_boxes))
        detection_scores.append(sample_detection_scores)
        detection_classes.append(sample_detection_classes)


    def load_groundtruth_boxes(nuscenes, sample_tokens):
        gt_box3ds = []

        # Load annotations and filter predictions and annotations.
        for sample_token in tqdm(sample_tokens):

            sample = nuscenes.get('sample', sample_token)
            sample_annotation_tokens = sample['anns']

            # sample_lidar_token = sample["data"]["LIDAR_TOP"]
            # lidar_data = level5data.get("sample_data", sample_lidar_token)
            # ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
            # ego_translation = np.array(ego_pose['translation'])

            for sample_annotation_token in sample_annotation_tokens:
                sample_annotation = nuscenes.get('sample_annotation', sample_annotation_token)
                sample_annotation_translation = sample_annotation['translation']

                class_name = sample_annotation['category_name']

                box3d = Box3D(
                    sample_token=sample_token,
                    translation=sample_annotation_translation,
                    size=sample_annotation['size'],
                    rotation=sample_annotation['rotation'],
                    name=class_name
                )
                gt_box3ds.append(box3d)

        return gt_box3ds

    gt_box3ds = load_groundtruth_boxes(level5data, sample_tokens)

    pred_box3ds = []

    bev_shape = (cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.IMG_CHANNELS)
    voxel_size = (cfg.VOXEL_SIZE, cfg.VOXEL_SIZE, cfg.VOXEL_Z_SIZE)
    z_offset = cfg.Z_OFFSET

    # This could use some refactoring..
    for (sample_token, sample_boxes, sample_detection_scores, sample_detection_class) in \
            tqdm(
                zip(sample_tokens, detection_boxes, detection_scores, detection_classes),
                total=len(sample_tokens)
            ):

        sample_boxes = sample_boxes.reshape(-1, 2)  # (N, 4, 2) -> (N*4, 2)
        sample_boxes = sample_boxes.transpose(1, 0)  # (N*4, 2) -> (2, N*4)

        # Add Z dimension
        sample_boxes = np.vstack((sample_boxes, np.zeros(sample_boxes.shape[1]),))  # (2, N*4) -> (3, N*4)

        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_voxel = np.linalg.inv(
            create_transformation_matrix_to_voxel_space(
                bev_shape,
                voxel_size,
                (0, 0, z_offset)
            )
        )

        global_from_voxel = np.dot(global_from_car, car_from_voxel)
        sample_boxes = transform_points(sample_boxes, global_from_voxel)

        # We don't know at where the boxes are in the scene on the z-axis (up-down), let's assume all of them are at
        # the same height as the ego vehicle.
        sample_boxes[2, :] = ego_pose["translation"][2]

        # (3, N*4) -> (N, 4, 3)
        sample_boxes = sample_boxes.transpose(1, 0).reshape(-1, 4, 3)

        sample_boxes_centers = sample_boxes.mean(axis=1)

        for i, class_name in enumerate(sample_detection_class):
            sample_boxes_centers[i, 2] += class_to_height[class_name] / 2

        sample_lengths = np.linalg.norm(sample_boxes[:, 0, :] - sample_boxes[:, 1, :], axis=1) * 1 / cfg.BOX_SCALE
        sample_widths = np.linalg.norm(sample_boxes[:, 1, :] - sample_boxes[:, 2, :], axis=1) * 1 / cfg.BOX_SCALE

        sample_boxes_dimensions = np.zeros_like(sample_boxes_centers)

        for i, class_name in enumerate(sample_detection_class):

            if class_name in ('motorcycle', 'bicycle', 'pedestrian', 'animal'):
                sample_boxes_dimensions[i, 0] += class_to_width[class_name]
                sample_boxes_dimensions[i, 1] += class_to_len[class_name]
            else:
                sample_boxes_dimensions[i, 0] += sample_widths[i]
                sample_boxes_dimensions[i, 1] += sample_lengths[i]

            sample_boxes_dimensions[i, 2] += class_to_height[class_name]

        for i in range(len(sample_boxes)):
            translation = sample_boxes_centers[i]
            size = sample_boxes_dimensions[i]
            class_name = sample_detection_class[i]
            ego_distance = float(np.linalg.norm(ego_translation - translation))

            # Determine the rotation of the box
            v = (sample_boxes[i, 0] - sample_boxes[i, 1])
            v /= np.linalg.norm(v)
            r = R.from_dcm([
                [v[0], -v[1], 0],
                [v[1], v[0], 0],
                [0, 0, 1],
            ])
            quat = r.as_quat()
            # XYZW -> WXYZ order of elements
            quat = quat[[3, 0, 1, 2]]

            detection_score = float(sample_detection_scores[i])

            box3d = Box3D(
                sample_token=sample_token,
                translation=list(translation),
                size=list(size),
                rotation=list(quat),
                name=class_name,
                score=detection_score
            )
            pred_box3ds.append(box3d)

    gts = [b.serialize() for b in gt_box3ds]
    predictions = [b.serialize() for b in pred_box3ds]

    print('gts len:', len(gts))
    print('predictions len:', len(predictions))

    local_predictions = predictions

    print('predictions len after:', len(local_predictions))

    mAPs = list()

    class_names = sorted(list(set([x['name'] for x in gts])))
    print(class_names)
    class_names = sorted(classes)
    print(class_names)
    print()

    precision_list = list()

    ious = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)

    with Pool(processes=10) as pool:
        stats = pool.map(
            partial(get_average_precisions, gts, local_predictions, class_names),
            ious
        )

    for average_precisions, iou_threshold in zip(stats, ious):
        mAP = np.mean(average_precisions)
        mAPs.append(mAP)
        print("Per class average precision at iou {:.2f} = {:.3f}".format(iou_threshold, mAP))

        for class_id in sorted(list(zip(class_names, average_precisions.flatten().tolist()))):
            print(class_id)
            precision_list.append(class_id)

        print('\n')

    print('mAP: {:.4f}\n\n\n'.format(sum(mAPs) / len(mAPs)))


if __name__ == '__main__':
    # for fold in (0, 1, 2, 3):
    for fold in (0,):
        p = Process(target=predict, args=(fold,))
        p.start()
        p.join()
