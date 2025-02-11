from functools import partial
from typing import Tuple

import torch
from torch import Tensor, nn

from collections import Sized, Iterable

import torch
from torch import Tensor


def torch_none(x: Tensor):
    return x


def torch_rot90_(x: Tensor):
    return x.transpose_(2, 3).flip(2)


def torch_rot90(x: Tensor):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x: Tensor):
    return x.flip(2).flip(3)


def torch_rot270(x: Tensor):
    return x.transpose(2, 3).flip(3)


def torch_flipud(x: Tensor):
    """
    Flip image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x: Tensor):
    """
    Flip image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)


def torch_transpose(x: Tensor):
    return x.transpose(2, 3)


def torch_transpose_(x: Tensor):
    return x.transpose_(2, 3)


def torch_transpose2(x: Tensor):
    return x.transpose(3, 2)


def pad_image_tensor(image_tensor: Tensor, pad_size: int = 32):
    """Pad input tensor to make it's height and width dividable by @pad_size

    :param image_tensor: Input tensor of shape NCHW
    :param pad_size: Pad size
    :return: Tuple of output tensor and pad params. Second argument can be used to reverse pad operation of model output
    """
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    if isinstance(pad_size, Sized) and isinstance(pad_size, Iterable) and len(pad_size) == 2:
        pad_height, pad_width = [int(val) for val in pad_size]
    elif isinstance(pad_size, int):
        pad_height = pad_width = pad_size
    else:
        raise ValueError(
            f"Unsupported pad_size: {pad_size}, must be either tuple(pad_rows,pad_cols) or single int scalar.")

    if rows > pad_height:
        pad_rows = rows % pad_height
        pad_rows = pad_height - pad_rows if pad_rows > 0 else 0
    else:
        pad_rows = pad_height - rows

    if cols > pad_width:
        pad_cols = cols % pad_width
        pad_cols = pad_width - pad_cols if pad_cols > 0 else 0
    else:
        pad_cols = pad_width - cols

    if pad_rows == 0 and pad_cols == 0:
        return image_tensor, (0, 0, 0, 0)

    pad_top = pad_rows // 2
    pad_btm = pad_rows - pad_top

    pad_left = pad_cols // 2
    pad_right = pad_cols - pad_left

    pad = [pad_left, pad_right, pad_top, pad_btm]
    image_tensor = torch.nn.functional.pad(image_tensor, pad)
    return image_tensor, pad


def unpad_image_tensor(image_tensor, pad):
    pad_left, pad_right, pad_top, pad_btm = pad
    rows, cols = image_tensor.size(2), image_tensor.size(3)
    return image_tensor[..., pad_top:rows - pad_btm, pad_left: cols - pad_right]


def unpad_xyxy_bboxes(bboxes_tensor: torch.Tensor, pad, dim=-1):
    pad_left, pad_right, pad_top, pad_btm = pad
    pad = torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=bboxes_tensor.dtype).to(bboxes_tensor.device)

    if dim == -1:
        dim = len(bboxes_tensor.size()) - 1

    expand_dims = list(set(range(len(bboxes_tensor.size()))) - {dim})
    for i, dim in enumerate(expand_dims):
        pad = pad.unsqueeze(dim)

    return bboxes_tensor - pad


def fliplr_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image segmentation that averages predictions
    for input image and vertically flipped one.

    For segmentation we need to reverse the transformation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)['logits']
    output_fliplr = model(torch_fliplr(image))['logits']

    output = output + torch_fliplr(output_fliplr)
    one_over_2 = float(1.0 / 2.0)
    return output * one_over_2


def d4_image2mask(model: nn.Module, image: Tensor) -> Tensor:
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.

    For segmentation we need to reverse the augmentation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image)

    for aug, deaug in zip([torch_rot90, torch_rot180, torch_rot270], [torch_rot270, torch_rot180, torch_rot90]):
        x = model(aug(image))
        x = deaug(x)
        output = output + x

    image = torch_transpose(image)

    for aug, deaug in zip([torch_none, torch_rot90, torch_rot180, torch_rot270],
                          [torch_none, torch_rot270, torch_rot180, torch_rot90]):
        x = model(aug(image))
        x = deaug(x)
        output = output + torch_transpose(x)

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8


class TTAWrapper(nn.Module):
    def __init__(self, model, tta_function, **kwargs):
        super().__init__()
        self.model = model
        self.tta = partial(tta_function, **kwargs)

    def forward(self, x):
        return {'logits': self.tta(self.model, x)}
