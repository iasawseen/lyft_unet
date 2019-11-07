import torch
import torch.nn as nn
from collections import OrderedDict
from catalyst.dl import SupervisedRunner
from configs.config import get_cfg
from data.dataset import get_dataloaders
from models.utils import get_unet_model
from utils.losses import FocalLoss
from utils.schedulers import CosineWithRestarts
from apex import amp
from models.utils import DataParallelCustom
from catalyst.contrib.criterion import DiceLoss, IoULoss
from catalyst.contrib.optimizers import RAdam, Lookahead

from catalyst.dl.callbacks import DiceCallback, IouCallback, \
  CriterionCallback, CriterionAggregatorCallback, SchedulerCallback

import os
os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':
    cfg = get_cfg()

    print('ARTIFACTS_FOLDER', cfg.ARTIFACTS_FOLDER)

    loaders = OrderedDict()

    train_loader, valid_loader = get_dataloaders(cfg)

    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    # experiment setup
    num_epochs = 128

    # logdir = "./logs/vehicles_v0.1"
    # logdir = "./logs/all_classes_v0.1"
    # logdir = "./logs/all_classes_v0.1_adjacent_clouds_bigger_lr"
    # logdir = "./logs/all_classes_v0.1_adjacent_clouds_as_channels"
    # logdir = "./logs/all_classes_v0.1_adjacent_clouds_as_channels_long"
    # logdir = "./logs/all_classes_v0.1_adjacent_clouds_as_channels_1280_5_frames"
    logdir = "./logs/small_classes_v0.1"

    model = get_unet_model(
        cfg.IMG_CHANNELS * 2 + 3,
        num_output_classes=10,
        backbone_name=cfg.BACKBONE,
        dropout=cfg.DROPOUT,
        input_dropout=cfg.INPUT_DROPOUT,
    )

    criterion = {
        "focal": FocalLoss(gamma=2.0),
        # "iou": IoULoss(),
        # "bce": nn.BCEWithLogitsLoss()
    }

    optimizer = RAdam(model.parameters_with_lrs(), lr=cfg.LR_MAX)

    scheduler = CosineWithRestarts(
        optimizer,
        cycle_len=cfg.CYCLE_LEN * len(train_loader),
        lr_min=cfg.LR_MIN,
        factor=1.4,
        gamma=0.85
    )

    model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=False)

    model = DataParallelCustom(model)

    # model runner
    runner = SupervisedRunner(input_key="image", input_target_key="mask")

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        fp16=True,
        callbacks=[
            # Each criterion is calculated separately.
            CriterionCallback(
                input_key="mask",
                prefix="loss_focal",
                criterion_key="focal"
            ),
            CriterionAggregatorCallback(
                prefix="loss",
                loss_keys=['loss_focal'],
                loss_aggregate_fn="sum"  # or "mean"
            ),
            SchedulerCallback(mode='batch')
            # DiceCallback(input_key="mask"),
            # IouCallback(input_key="mask"),
        ]
    )

