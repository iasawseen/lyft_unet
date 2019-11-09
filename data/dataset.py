import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import sparse
from datetime import datetime
import glob
from copy import deepcopy

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.

import os
os.environ["OMP_NUM_THREADS"] = "1"


import pandas as pd
import cv2
import numpy as np

from lyft_dataset_sdk.lyftdataset import LyftDataset

from data.augs import Augmenter
from data.generator import generate_data
from data.stats import get_class_stats, get_classes

import warnings
warnings.filterwarnings("ignore")


class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg,
                 input_filepaths, target_filepaths,
                 map_filepaths=None, target_wlh_filepaths=None, enable_aug=False, testing=False):

        self.cfg = cfg
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.target_wlh_filepaths = target_wlh_filepaths
        self.map_filepaths = map_filepaths
        self.enable_aug = enable_aug
        self.testing = testing

        print('input_filepaths', len(input_filepaths))
        print('target_filepaths', len(target_filepaths))
        # print('target_wlh_filepaths', len(target_wlh_filepaths))
        print('map_filepaths', len(map_filepaths))

        self.aug = Augmenter(input_img_size=cfg.IMG_SIZE, crop_img_size=cfg.IMG_SIZE_CROP)

        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)

        if not self.testing:
            assert len(input_filepaths) == len(target_filepaths)

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]
        im = sparse.load_npz(input_filepath).todense().astype(np.uint8)

        sample_token = input_filepath.split("/")[-1].replace("_input.npz", "")

        if self.map_filepaths:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
            im = np.concatenate((im, map_im), axis=2)

        if not self.testing:
            target_filepath = self.target_filepaths[idx]
            target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

        if self.target_wlh_filepaths is not None and not self.testing:
            target_wlh_filepath = self.target_wlh_filepaths[idx]
            target_wlh = sparse.load_npz(target_wlh_filepath).todense().astype(np.float32)

            target_wlh_ave = target_wlh[:, :, 3:]
            target_wlh_gt = target_wlh[:, :, :3]

            target_wlh_norm = (target_wlh_gt - target_wlh_ave) / target_wlh_ave

        if self.enable_aug:
            if self.target_wlh_filepaths is not None:
                im, [target, target_wlh_norm] = self.aug(im, [target, target_wlh_norm])
            else:
                im, [target] = self.aug(im, [target])

        im = im.astype(np.float32) / 255
        im = torch.from_numpy(im.transpose(2, 0, 1))

        if self.testing:
            return {'image': im, 'token': sample_token}

        target = target.astype(np.int64)
        target = torch.from_numpy(target)

        result = {'image': im, 'mask': target, 'token': sample_token}

        if self.target_wlh_filepaths is not None:
            # half_range = self.cfg.REG_HALF_RANGE
            # offset = self.cfg.REG_OFFSET
            # target_wlh_norm += half_range
            # target_wlh_norm = np.clip(target_wlh_norm, 0.0, half_range * 2)
            # target_wlh_norm /= offset
            # target_wlh_norm = target_wlh_norm.astype(np.long)
            target_wlh_norm = torch.from_numpy(target_wlh_norm.transpose(2, 0, 1))
            result['wlh'] = target_wlh_norm

        return result


def get_dataloader(cfg, file_paths, ratio=1.0, train=False, num_workers=4, testing=False):
    input_file_paths, target_file_paths, target_wlh_file_paths, map_file_paths = \
        file_paths['input'], file_paths['target'], file_paths['target_wlh'], file_paths['map']

    length = int(len(input_file_paths) * ratio)

    input_file_paths = input_file_paths[:length]
    target_file_paths = target_file_paths[:length]
    target_wlh_file_paths = target_wlh_file_paths[:length]
    map_file_paths = map_file_paths[:length]

    dataset = BEVImageDataset(
        cfg,
        input_file_paths,
        target_file_paths,
        map_filepaths=map_file_paths,
        target_wlh_filepaths=target_wlh_file_paths if cfg.REGRESSION else None,
        enable_aug=train,
        testing=testing
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, cfg.BATCH_SIZE // (1 if train else 4), shuffle=train, num_workers=num_workers
    )

    return dataloader


def get_tokens(level5data, df):
    tokens = set()
    first_sample_tokens = df.first_sample_token.values

    for first_sample_token in first_sample_tokens:
        sample_token = first_sample_token
        while sample_token:
            tokens.add(sample_token)
            sample = level5data.get("sample", sample_token)
            sample_token = sample["next"]

    return tokens


def get_file_paths(data_folder):
    input_filepaths = list(sorted(glob.glob(os.path.join(data_folder, "*_input.npz"))))
    target_filepaths = list(sorted(glob.glob(os.path.join(data_folder, "*_target.png"))))
    target_wlh_filepaths = list(sorted(glob.glob(os.path.join(data_folder, "*_target_wlh.npz"))))
    map_filepaths = list(sorted(glob.glob(os.path.join(data_folder, "*_map.png"))))

    # print('input_filepaths', len(input_filepaths))
    # print('target_filepaths', len(target_filepaths))
    # print('target_wlh_filepaths', len(target_wlh_filepaths))
    # print('map_filepaths', len(map_filepaths))

    return {
        'input': input_filepaths,
        'target': target_filepaths,
        'target_wlh': target_wlh_filepaths,
        'map': map_filepaths
    }


def get_input_file_paths(train_data_folder, val_data_folder):
    train_file_paths = get_file_paths(train_data_folder)
    val_file_paths = get_file_paths(val_data_folder)

    for key in train_file_paths:
        # print('file_paths before: {}'.format(len(train_file_paths[key])))
        train_file_paths[key].extend(val_file_paths[key])
        # print('file_paths after: {}'.format(len(train_file_paths[key])))
        # print()

    return train_file_paths


def filter_file_paths(file_paths, tokens):
    def file_path_in_tokens(file_path, tokens):
        return file_path.split('/')[-1].split('_')[0] in tokens

    for key in file_paths:
        # print('key: {}, len before: {}'.format(key, len(file_paths[key])))
        file_paths[key] = list(sorted(filter(lambda x: file_path_in_tokens(x, tokens), file_paths[key])))
        # print('key: {}, len after: {}'.format(key, len(file_paths[key])))
        # print()

    return file_paths


def get_dataloaders(cfg, fold=0, val_ratio=1.0):
    level5data = LyftDataset(
        json_path=cfg.DATASET_ROOT + "/train_data/",
        data_path=cfg.DATASET_ROOT,
        verbose=False
    )

    os.makedirs(cfg.ARTIFACTS_FOLDER, exist_ok=True)

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

    fold_to_val_hosts = {
        0: ['host-a007', 'host-a008', 'host-a009'],
        1: ['host-a004', 'host-a005', 'host-a006'],
        2: ['host-a011', 'host-a012', 'host-a015'],
        3: ['host-a017', 'host-a101', 'host-a102']
    }

    validation_hosts = fold_to_val_hosts[fold]

    val_df = df[df["host"].isin(validation_hosts)]
    vi = val_df.index
    train_df = df[~df.index.isin(vi)]

    train_data_folder = os.path.join(cfg.ARTIFACTS_FOLDER, "bev_train_data")
    val_data_folder = os.path.join(cfg.ARTIFACTS_FOLDER, "./bev_validation_data")

    train_tokens = get_tokens(level5data, train_df)
    val_tokens = get_tokens(level5data, val_df)

    file_paths = get_input_file_paths(train_data_folder, val_data_folder)

    train_file_paths = filter_file_paths(deepcopy(file_paths), train_tokens)
    val_file_paths = filter_file_paths(deepcopy(file_paths), val_tokens)

    train_dataloader = get_dataloader(
        cfg, train_file_paths, ratio=1.0, train=True, num_workers=12
    )
    validation_dataloader = get_dataloader(
        cfg, val_file_paths, ratio=val_ratio, train=False, num_workers=6
    )

    return train_dataloader, validation_dataloader


def get_test_dataloader(cfg, ratio=1.0):
    level5data = LyftDataset(
        json_path=cfg.DATASET_ROOT + "/test_data/",
        data_path=cfg.DATASET_ROOT,
        verbose=False
    )

    os.makedirs(cfg.ARTIFACTS_FOLDER, exist_ok=True)

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

    test_data_folder = os.path.join(cfg.ARTIFACTS_FOLDER, "bev_test_data")

    print('test_data_folder', test_data_folder)

    test_file_paths = get_file_paths(test_data_folder)

    for key in test_file_paths:
        print(key, len(test_file_paths[key]))
    print()

    test_dataloader = get_dataloader(
        cfg, test_file_paths, ratio=ratio, train=False, num_workers=12, testing=True
    )

    return test_dataloader
