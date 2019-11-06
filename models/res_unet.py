from torch import nn
from torch.nn import functional as F
import torch
import pretrainedmodels



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out, stride):
        super().__init__()
        self.conv = conv3x3(in_, out, stride)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 3, 5), batch_norm=False):
        super().__init__()
        self.conv_0 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
        self.batch_norm = batch_norm

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation),
                nn.ReLU(inplace=True)
            ) for dilation in dilations]
        )

    def forward(self, x):
        conv0 = self.conv_0(x)
        conv0 = F.relu(conv0, inplace=True)
        blocks = [self.blocks[index](conv0) for index in range(len(self.blocks))]
        return torch.cat(blocks, dim=1)


class SawSeenUnet(nn.Module):
    def __init__(self, input_channels, num_classes, backbone_name='se_resnext50_32x4d', base_channels=64, dropout=0.3):
        super(SawSeenUnet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.model_name = backbone_name
        self.base_channels = base_channels

        self.embedding_net = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')

        if self.model_name in ('resnet34', 'resnet50'):
            self.init_conv = self.embedding_net.conv1
            self.bn1 = self.embedding_net.bn1
            self.relu = self.embedding_net.relu
            self.maxpool = self.embedding_net.maxpool
        else:
            self.init_conv = self.embedding_net.layer0.conv1
            self.bn1 = self.embedding_net.layer0.bn1
            self.relu = self.embedding_net.layer0.relu1
            self.maxpool = self.embedding_net.layer0.pool

        self.enc_0 = self.embedding_net.layer1
        self.enc_1 = self.embedding_net.layer2
        self.enc_2 = self.embedding_net.layer3
        self.enc_3 = self.embedding_net.layer4

        middle_in_channels = 512 if self.model_name == 'resnet34' else self.base_channels * 32

        self.middle_conv = ConvRelu(middle_in_channels, self.base_channels * 8, stride=2)

        self.dec_3 = DecoderBlock(self.base_channels * 8, self.base_channels * 4, dilations=(1,))

        dec_2_in_channels = 768 if self.model_name == 'resnet34' else 2304

        self.dec_2 = DecoderBlock(dec_2_in_channels, self.base_channels * 2, dilations=(3,))

        dec_1_in_channels = dec_2_in_channels // 2

        self.dec_1 = DecoderBlock(dec_1_in_channels, self.base_channels * 1, dilations=(3,))

        dec_0_in_channels = 192 if self.model_name == 'resnet34' else 576

        self.dec_0 = DecoderBlock(dec_0_in_channels, self.base_channels // 2, dilations=(5,))

        dec_final_0_in_channels = 96 if self.model_name == 'resnet34' else 288

        self.dec_final_0 = DecoderBlock(dec_final_0_in_channels, self.base_channels // 2, dilations=(5,))

        self.dropout = nn.Dropout(p=dropout)

        self.final = nn.Conv2d(32, self.num_classes, kernel_size=5, padding=2)

        init_conv_weight = self.init_conv.weight

        self.init_conv = nn.Conv2d(
            self.input_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # noinspection PyArgumentList
        self.init_conv.weight = torch.nn.Parameter(
            torch.cat(
                [init_conv_weight for _ in range(self.input_channels // 3)], dim=1
            )
        )

    def forward(self, x):
        init_conv = self.init_conv(x)
        init_conv = self.bn1(init_conv)
        init_conv = self.relu(init_conv)

        enc_0 = self.enc_0(init_conv)
        enc_1 = self.enc_1(enc_0)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)

        middle_conv = self.middle_conv(enc_3)

        dec_3 = self.dec_3(middle_conv)

        dec_3_cat = torch.cat([
            dec_3,
            enc_3
        ], 1)

        dec_2 = self.dec_2(dec_3_cat)

        dec_2_cat = torch.cat([
            dec_2,
            enc_2
        ], 1)

        dec_1 = self.dec_1(dec_2_cat)

        dec_1_cat = torch.cat([
            dec_1,
            enc_1
        ], 1)

        dec_0 = self.dec_0(dec_1_cat)

        dec_0_cat = torch.cat([
            dec_0,
            enc_0
        ], 1)

        dec_final_0 = self.dec_final_0(dec_0_cat)

        dec_final_0 = self.dropout(dec_final_0)

        final = self.final(dec_final_0)

        return final

    def parameters_with_lrs(self, lrs=(0.1, 0.3, 1.0)):
        return [
            {'params': self.init_conv.parameters(),    'lr_factor': lrs[0]},
            {'params': self.bn1.parameters(),          'lr_factor': lrs[0]},
            {'params': self.enc_0.parameters(),        'lr_factor': lrs[1]},
            {'params': self.enc_1.parameters(),        'lr_factor': lrs[1]},
            {'params': self.enc_2.parameters(),        'lr_factor': lrs[1]},
            {'params': self.enc_3.parameters(),        'lr_factor': lrs[1]},

            {'params': self.middle_conv.parameters(),  'lr_factor': lrs[2]},
            {'params': self.dec_3.parameters(),        'lr_factor': lrs[2]},
            {'params': self.dec_2.parameters(),        'lr_factor': lrs[2]},
            {'params': self.dec_1.parameters(),        'lr_factor': lrs[2]},
            {'params': self.dec_0.parameters(),        'lr_factor': lrs[2]},
            {'params': self.dec_final_0.parameters(),  'lr_factor': lrs[2]},
            {'params': self.final.parameters(),        'lr_factor': lrs[2]},
        ]


class SawSeenUberUnet(nn.Module):
    def __init__(self,
                 input_channels, num_classes, num_reg=None,
                 backbone_name='se_resnext50_32x4d',
                 base_channels=64, dropout=0.2):

        super(SawSeenUberUnet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_reg = num_reg

        self.model_name = backbone_name
        self.base_channels = base_channels

        self.embedding_net = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')

        if self.model_name in ('resnet18', 'resnet34', 'resnet50'):
            self.init_conv = self.embedding_net.conv1
            self.bn1 = self.embedding_net.bn1
            self.relu = self.embedding_net.relu
            self.maxpool = self.embedding_net.maxpool
        else:
            self.init_conv = self.embedding_net.layer0.conv1
            self.bn1 = self.embedding_net.layer0.bn1
            self.relu = self.embedding_net.layer0.relu1
            self.maxpool = self.embedding_net.layer0.pool

        self.enc_0 = self.embedding_net.layer1
        self.enc_1 = self.embedding_net.layer2
        self.enc_2 = self.embedding_net.layer3
        self.enc_3 = self.embedding_net.layer4

        small_resnets = ('resnet18', 'resnet34')

        middle_in_channels = 512 if self.model_name in small_resnets else self.base_channels * 32

        self.middle_conv = ConvRelu(middle_in_channels, self.base_channels * 8, stride=2)

        self.dec_3 = DecoderBlock(self.base_channels * 8, self.base_channels * 4, dilations=(1,))

        dec_2_in_channels = 768 if self.model_name in small_resnets else 2304

        self.dec_2 = DecoderBlock(dec_2_in_channels, self.base_channels * 2, dilations=(1, 3,))

        dec_1_in_channels = 512

        self.dec_1 = DecoderBlock(dec_1_in_channels, self.base_channels * 1, dilations=(1, 3,))

        dec_0_in_channels = 256 if self.model_name in small_resnets else 576

        self.dec_0 = DecoderBlock(dec_0_in_channels, self.base_channels // 2, dilations=(1, 3, 5,))

        dec_final_0_in_channels = 160 if self.model_name in small_resnets else 288

        self.dec_final_0 = DecoderBlock(dec_final_0_in_channels, self.base_channels // 2, dilations=(1, 3, 5,))

        self.dropout = nn.Dropout2d(p=dropout)

        self.final = nn.Conv2d(96, self.num_classes, kernel_size=3, padding=1)

        if self.num_reg is not None:
            self.reg = nn.Conv2d(96, self.num_reg, kernel_size=3, padding=1)

        init_conv_weight = self.init_conv.weight

        self.init_conv = nn.Conv2d(
            self.input_channels, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # noinspection PyArgumentList
        self.init_conv.weight = torch.nn.Parameter(
            torch.cat(
                [init_conv_weight for _ in range(self.input_channels // 3)], dim=1
            )
        )

    def forward(self, x):
        init_conv = self.init_conv(x)
        init_conv = self.bn1(init_conv)
        init_conv = self.relu(init_conv)

        enc_0 = self.enc_0(init_conv)
        enc_1 = self.enc_1(enc_0)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)

        middle_conv = self.middle_conv(enc_3)

        dec_3 = self.dec_3(middle_conv)

        dec_3_cat = torch.cat([
            dec_3,
            enc_3
        ], 1)

        dec_2 = self.dec_2(dec_3_cat)

        dec_2_cat = torch.cat([
            dec_2,
            enc_2
        ], 1)

        dec_1 = self.dec_1(dec_2_cat)

        dec_1_cat = torch.cat([
            dec_1,
            enc_1
        ], 1)

        dec_0 = self.dec_0(dec_1_cat)

        dec_0_cat = torch.cat([
            dec_0,
            enc_0
        ], 1)

        dec_final_0 = self.dec_final_0(dec_0_cat)

        dec_final_0 = self.dropout(dec_final_0)

        final = self.final(dec_final_0)

        if self.num_reg is not None:
            final_reg = self.reg(dec_final_0)
            return final, final_reg

        return final

    def parameters_with_lrs(self, lrs=(0.1, 0.3, 1.0)):
        params = [
            {'params': self.init_conv.parameters(), 'lr_factor': lrs[0]},
            {'params': self.bn1.parameters(), 'lr_factor': lrs[0]},
            {'params': self.enc_0.parameters(), 'lr_factor': lrs[1]},
            {'params': self.enc_1.parameters(), 'lr_factor': lrs[1]},
            {'params': self.enc_2.parameters(), 'lr_factor': lrs[1]},
            {'params': self.enc_3.parameters(), 'lr_factor': lrs[1]},

            {'params': self.middle_conv.parameters(), 'lr_factor': lrs[2]},
            {'params': self.dec_3.parameters(), 'lr_factor': lrs[2]},
            {'params': self.dec_2.parameters(), 'lr_factor': lrs[2]},
            {'params': self.dec_1.parameters(), 'lr_factor': lrs[2]},
            {'params': self.dec_0.parameters(), 'lr_factor': lrs[2]},
            {'params': self.dec_final_0.parameters(), 'lr_factor': lrs[2]},
            {'params': self.final.parameters(), 'lr_factor': lrs[2]},
        ]

        if self.num_reg is not None:
            params.append(
                {'params': self.reg.parameters(), 'lr_factor': lrs[2]},
            )

        return params


def _requires_grad_model(model, requires_grad):

    encoder_layers = (
        model.init_conv,
        model.bn1,
        model.enc_0,
        model.enc_1,
        model.enc_2,
        model.enc_3,
    )

    for layer in encoder_layers:
        for p in layer.parameters():
            p.requires_grad_(requires_grad)


def freeze_unet(model):
    _requires_grad_model(model, requires_grad=False)


def unfreeze_unet(model):
    _requires_grad_model(model, requires_grad=True)


INPUT_CHANNELS = 18
NUM_CLASSES = 10


if __name__ == '__main__':
    segmentator = SawSeenUberUnet(
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=64,
        backbone_name='resnet18'
        # backbone_name='resnet34'
        # backbone_name='se_resnet50'
        # backbone_name='resnet50'
    )

    # freeze_unet(segmentor)

    pic = torch.randn(1, INPUT_CHANNELS, 256, 256)
    result, result_reg = segmentator(pic)
    print(result.size(), result_reg.size())

    optimizer = torch.optim.Adam(
        segmentator.parameters_with_lrs(),
        lr=5e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0001
    )
