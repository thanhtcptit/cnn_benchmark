import os
import cv2
import numpy as np
import tensorpack as tp

from tensorpack.dataflow import imgaug


def parse_augmentor(augmentors_params, is_train):
    # TODO load augmentors in-order with the config file
    augmentors_params = dict(augmentors_params)
    if is_train:
        augmentors_params = augmentors_params['training']
    else:
        augmentors_params = augmentors_params['test']

    augmentors = []
    for name, params in augmentors_params.items():
        params = dict(params)
        aug_class = getattr(imgaug, name)
        aug_inst = aug_class(**params)
        augmentors.append(aug_inst)
    return augmentors


def fbresnet_augmentor(is_train):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    interpolation = cv2.INTER_LINEAR
    if is_train:
        augmentors = [
            imgaug.GoogleNetRandomCropAndResize(interp=interpolation),
            # It's OK to remove the following augs if your CPU is slow.
            # Removing brightness/contrast/saturation does not have
            # a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from
                 # fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, interp=interpolation),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def standard_augmentor(is_train):
    if is_train:
        augmentors = [
            imgaug.Rotation(10),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    return augmentors


def flip_rotate_normalize(is_train):
    if is_train:
        augmentors = [
            imgaug.Rotation(10),
            imgaug.Flip(horiz=True),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    return augmentors


def identity_augmentor():
    return [imgaug.Identity()]
