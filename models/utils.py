import torch.nn as nn
from models.res_unet import SawSeenUberUnet


class DataParallelCustom(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelCustom, self).__init__(module, device_ids, output_device, dim)

    def parameters_with_lrs(self, lrs=(0.1, 0.3, 1.0)):
        parameters = self.module.parameters_with_lrs(lrs)
        return parameters


def get_unet_model(in_channels=3, num_output_classes=2, backbone_name='resnet34', dropout=0.0, input_dropout=0.0):
    model = SawSeenUberUnet(
        input_channels=in_channels,
        num_classes=num_output_classes,
        backbone_name=backbone_name,
        dropout=dropout,
        input_dropout=input_dropout
    )

    return model
