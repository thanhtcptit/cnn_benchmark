import os
import cv2
import nsds
import numpy as np

from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from tensorpack.dataflow import RNGDataFlow, AugmentImageComponent, \
    BatchData, MultiThreadMapData, PrefetchDataZMQ, dataset, imgaug
from tensorpack.input_source import QueueInput, StagingInput, TFDatasetInput

from cnn_benchmark.data.image_aug import parse_augmentor, standard_augmentor, \
    fbresnet_augmentor, flip_rotate_normalize, normalize, identity_augmentor


def get_image_paths_and_labels(data_dir):
    img_paths = []
    img_labels = []

    labels = sorted(os.listdir(data_dir))
    for i, label in enumerate(labels):
        label_dir = os.path.join(data_dir, label)
        img_list = os.listdir(label_dir)
        if len(img_list) < 2:
            print(f'Class "{label}" has less than 2 element. Skip')
            continue
        img_paths.extend([os.path.join(label_dir, img)
                         for img in img_list])
        img_labels.extend([i] * len(img_list))
    return img_paths, img_labels


class ImageFileDataflow(RNGDataFlow):
    def __init__(self, img_paths, img_labels, shuffle=False):
        self.img_paths, self.img_labels = img_paths, img_labels
        self._shuffle = shuffle

    def __len__(self):
        return len(self.img_labels)

    def __iter__(self):
        idxs = np.arange(len(self.img_labels))
        if self._shuffle:
            self.rng.shuffle(idxs)
        for i in idxs:
            yield self.img_paths[i], self.img_labels[i]


class ImageDataflow(ImageFileDataflow):
    def __init__(self, img_paths, img_labels, shuffle=False):
        super(ImageDataflow, self).__init__(img_paths, img_labels, shuffle)

    def __iter__(self):
        for fpath, label in super(ImageDataflow, self).__iter__():
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            assert img is not None, fpath
            yield img, label


def get_image_dataflow(data_dir, batch_size, num_parallel,
                       augmentor_params=None, validation_split_ratio=None):
    """Helper function by auto detect data structure and
    build appropriate dataflow
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_img_paths, train_labels = get_image_paths_and_labels(train_dir)
        val_img_paths, val_labels = get_image_paths_and_labels(val_dir)
    else:
        assert len(os.listdir(data_dir)) >= 2, 'Number of class must be >= 2'
        assert validation_split_ratio, "Validation split ratio is None"
        assert 0. < validation_split_ratio < 1., \
            f"Invalid validation split ratio ({validation_split_ratio})"

        img_paths, img_labels = get_image_paths_and_labels(data_dir)
        train_img_paths, val_img_paths, train_labels, val_labels = \
            train_test_split(img_paths, img_labels,
                             test_size=validation_split_ratio,
                             stratify=img_labels)

    train_ds = build_dataflow(train_img_paths, train_labels, True, batch_size,
                              num_parallel, augmentor_params)
    val_ds = build_dataflow(val_img_paths, val_labels, False, batch_size,
                            num_parallel, augmentor_params)
    return train_ds, val_ds


def build_dataflow(img_paths, img_labels, is_train, batch_size,
                   num_parallel=None, augmentor_params=None):
    if num_parallel is None:
        num_parallel = min(40, cpu_count() // 2)
    if isinstance(augmentor_params, str):
        if augmentor_params == 'standard':
            augmentors = standard_augmentor(is_train)
        elif augmentor_params == 'fbresnet':
            augmentors = fbresnet_augmentor(is_train)
        elif augmentor_params == 'normalize':
            augmentors = normalize()
        elif augmentor_params == 'flip_rotate_normalize':
            augmentors = flip_rotate_normalize(is_train)
        else:
            raise ValueError('Unrecognize augmentor option')
    elif isinstance(augmentor_params, nsds.common.params.Params):
        augmentors = parse_augmentor(augmentor_params, is_train)
    elif augmentor_params is None:
        augmentors = identity_augmentor()

    if is_train:
        ds = ImageDataflow(img_paths, img_labels, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = PrefetchDataZMQ(ds, num_proc=num_parallel)
        ds = BatchData(ds, batch_size=batch_size, remainder=False)
    else:
        ds = ImageFileDataflow(img_paths, img_labels)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fpath, label = dp
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            return aug.augment(img), label

        ds = MultiThreadMapData(ds, num_thread=num_parallel, map_func=mapf,
                                buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size=batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, num_proc=1)
    return QueueInput(ds)


def get_cifar_dataflow(train_or_test, cifar_classnum, augmentors=None):
    is_train = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    if augmentors is None:
        augmentors = cifar_augmentor(is_train)
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not is_train)
    if is_train:
        ds = PrefetchDataZMQ(ds, 5)
    return QueueInput(ds)
