import torch
import torch.nn as nn
from collections import OrderedDict
from catalyst.dl import SupervisedRunner
from configs.config import get_cfg
from data.dataset import get_dataloaders
from models.utils import get_unet_model
from utils.losses import FocalLoss, SegmentationFocalLoss, RegressionFocalLoss, RegressionLoss
from utils.schedulers import CosineWithRestarts
from apex import amp
from models.utils import DataParallelCustom
from catalyst.contrib.criterion import DiceLoss, IoULoss
from catalyst.contrib.optimizers import RAdam, Lookahead

from catalyst.dl.callbacks import DiceCallback, IouCallback, \
  CriterionCallback, CriterionAggregatorCallback, SchedulerCallback

from multiprocessing import Process

import os
os.environ["OMP_NUM_THREADS"] = "1"


def train(fold=0):
    print('Fold: {}'.format(fold))

    cfg = get_cfg()

    print('ARTIFACTS_FOLDER', cfg.ARTIFACTS_FOLDER)

    loaders = OrderedDict()

    train_loader, valid_loader = get_dataloaders(cfg, fold=fold)

    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    num_epochs = 21

    # logdir = "./logs/all_classes_v0.3_drop_0.25_{fold}".format(fold=fold)
    # logdir = "./logs/testing_all_classes_v0.3_drop_0.25_fold_{fold}".format(fold=fold)
    logdir = "./logs/all_classes_v0.3_drop_0.33_fold_{fold}".format(fold=fold)

    model = get_unet_model(
        cfg,
        cfg.IMG_CHANNELS * 2 + 3,
        num_output_classes=10,
        backbone_name=cfg.BACKBONE,
        dropout=cfg.DROPOUT,
        input_dropout=cfg.INPUT_DROPOUT,
    )

    criterion = {
        "focal": SegmentationFocalLoss(gamma=2.0),
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
            CriterionCallback(
                input_key="mask",
                prefix="loss_focal",
                criterion_key="focal"
            ),
            # CriterionCallback(
            #     input_key=["image", "mask", 'wlh'],
            #     prefix="loss_focal_reg",
            #     criterion_key="focal_reg",
            #     multiplier=0.4
            # ),
            CriterionAggregatorCallback(
                prefix="loss",
                loss_keys=['loss_focal'],
                loss_aggregate_fn="sum"  # or "mean"
            ),
            SchedulerCallback(mode='batch')
        ]
    )


if __name__ == '__main__':

    for fold in (0, 1, 2, 3):
        p = Process(target=train, args=(fold,))
        p.start()
        p.join()
