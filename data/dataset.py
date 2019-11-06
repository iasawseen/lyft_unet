import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import sparse
from datetime import datetime
import glob

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
                 map_filepaths=None, enable_aug=False):

        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths

        self.map_filepaths = map_filepaths
        self.enable_aug = enable_aug
        self.aug = Augmenter(input_img_size=cfg.IMG_SIZE, crop_img_size=cfg.IMG_SIZE_CROP)

        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)

        assert len(input_filepaths) == len(target_filepaths)

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]
        target_filepath = self.target_filepaths[idx]
        # target_wlh_filepath = self.target_wlh_filepaths[idx]

        sample_token = target_filepath.split("/")[-1].replace("_target.png", "")

        # im = np.load(input_filepath)

        im = sparse.load_npz(input_filepath).todense().astype(np.uint8)
        # target_wlh = sparse.load_npz(target_wlh_filepath).todense().astype(np.float32)

        if self.map_filepaths:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
            im = np.concatenate((im, map_im), axis=2)

        target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)

        # target_wlh_ave = target_wlh[:, :, 3:]
        # target_wlh_gt = target_wlh[:, :, :3]

        # target_whl_norm = (target_wlh_gt - target_wlh_ave) / target_wlh_ave

        if self.enable_aug:
            im, [target] = self.aug(im, [target])

        im = im.astype(np.float32) / 255
        target = target.astype(np.int64)

        im = torch.from_numpy(im.transpose(2, 0, 1))
        # target_whl_norm = torch.from_numpy(target_whl_norm.transpose(2, 0, 1))

        target = torch.from_numpy(target)
        # return im, target, target_whl_norm, sample_token
        return {'image': im, 'mask': target}


def get_dataloader(data_folder):
    val_input_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_input.npz")))
    # val_target_wlh_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_target_wlh.npz")))
    val_target_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_target.png")))
    val_map_filepaths = sorted(glob.glob(os.path.join(data_folder, "*_map.png")))

    length = len(val_target_filepaths) // 4

    val_input_filepaths = val_input_filepaths[:length]
    val_target_filepaths = val_target_filepaths[:length]
    val_map_filepaths = val_map_filepaths[:length]

    validation_dataset = BEVImageDataset(
        cfg,
        val_input_filepaths, val_target_filepaths,
        val_map_filepaths
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, cfg.BATCH_SIZE // 4, shuffle=False, num_workers=4
    )



def get_dataloaders(cfg):
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
    host_count_df = df.groupby("host")['scene_token'].count()

    validation_hosts = ["host-a007", "host-a008", "host-a009"]

    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]

    train_data_folder = os.path.join(cfg.ARTIFACTS_FOLDER, "bev_train_data")
    validation_data_folder = os.path.join(cfg.ARTIFACTS_FOLDER, "./bev_validation_data")

    classes = get_classes()
    class_to_stats = get_class_stats()

    if cfg.DATASET_GENERATION:
        generate_data(
            cfg,
            level5data=level5data,
            train=(train_df, train_data_folder),
            val=(validation_df, validation_data_folder),
            classes=classes,
            class_to_stats=class_to_stats
        )

    length = 1000

    input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.npz")))
    # target_wlh_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target_wlh.npz")))
    target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))
    map_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_map.png")))


    train_dataset = BEVImageDataset(
        cfg,
        input_filepaths, target_filepaths,
        map_filepaths, enable_aug=True)

    dataloader = torch.utils.data.DataLoader(
        train_dataset, cfg.BATCH_SIZE, shuffle=True, num_workers=8)

    val_input_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_input.npz")))
    # val_target_wlh_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_target_wlh.npz")))
    val_target_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_target.png")))
    val_map_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_map.png")))

    length = len(val_target_filepaths) // 4

    val_input_filepaths = val_input_filepaths[:length]
    val_target_filepaths = val_target_filepaths[:length]
    val_map_filepaths = val_map_filepaths[:length]

    validation_dataset = BEVImageDataset(
        cfg,
        val_input_filepaths, val_target_filepaths,
        val_map_filepaths
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, cfg.BATCH_SIZE // 4, shuffle=False, num_workers=4
    )

    return dataloader, validation_dataloader

