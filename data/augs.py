import cv2

from albumentations.core.transforms_interface import ImageOnlyTransform

from albumentations import (
    IAAAffine, PadIfNeeded, HorizontalFlip, Rotate, VerticalFlip,
    RandomSizedCrop, CenterCrop, Crop, Compose, Cutout, CoarseDropout,
    Transpose, RandomRotate90, ElasticTransform, GridDistortion, OpticalDistortion,
    RandomSizedCrop, ShiftScaleRotate, OneOf, CLAHE, RandomContrast,
    RandomGamma, RandomBrightness, RandomBrightnessContrast, Blur, GaussNoise,
    ChannelShuffle, InvertImg, MotionBlur, RandomCrop
)

import warnings
warnings.filterwarnings("ignore")


class Augmenter:
    def __init__(self, input_img_size, crop_img_size):
        self.aug = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Rotate(limit=18, interpolation=cv2.INTER_NEAREST, p=0.5),
            Cutout(
                num_holes=128,
                max_h_size=int(16 * input_img_size / 384),
                max_w_size=int(16 * input_img_size / 384), p=0.5
            ),
            RandomCrop(height=crop_img_size, width=crop_img_size)
        ])

    def __call__(self, image, masks):
        augmented = self.aug(image=image, masks=masks)

        image_augmented = augmented['image']
        masks_augmented = augmented['masks']

        return image_augmented, masks_augmented
