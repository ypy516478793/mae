# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def init_imagenet(is_train, data_path, transform):
    root = os.path.join(data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

def init_imagenet_limit(is_train, data_path, transform, num_tr, num_val, num_cls):
    from .datasets_imagenet import ImageFolder_limit
    root = os.path.join(data_path, "train")
    dataset = ImageFolder_limit(root, is_train, num_tr, num_val, num_cls, transform=transform)
    return dataset

def init_imagenet_limit_with_name(is_train, data_path, transform, num_tr, num_val, num_cls):
    from .datasets_imagenet import ImageFolder_limit_with_name
    root = os.path.join(data_path, "train")
    dataset = ImageFolder_limit_with_name(root, is_train, num_tr, num_val, num_cls, transform=transform)
    return dataset

def init_imagenet_limit_use_features(is_train, data_path, transform):
    from .datasets_features import Features
    root = os.path.join(data_path, "train" if is_train else "test")
    dataset = Features(root, transform=transform)
    return dataset

def init_lung():
    from .datasets_lung import LungBoth
    luna_dir = "./LUNA16/preprocessed/"
    meth_dir = "./Methodist_incidental/data_Ben/preprocessed_data_v1"
    dataset = LungBoth(luna_dir, meth_dir)
    return dataset

def init_luna_nodule(is_train):
    from .datasets_lung import Luna_nodule
    root_dir = "./LUNA16/cls/crop_v6"
    label_path = "./LUNA16/cls/annotationdetclsconvfnl_v3.csv"
    split = "trainVal" if is_train else "test"
    dataset = Luna_nodule(root_dir, label_path, split)
    return dataset

def init_meth_nodule(is_train):
    from .datasets_lung import Methodist_nodule
    split = "trainVal" if is_train else "test"
    root_dir = "./Methodist_incidental/Nodules/patch32"
    label_path = "./Methodist_incidental/Nodules/patch32/cls_labels.csv"
    dataset = Methodist_nodule(root_dir, label_path, split)
    return dataset

def init_lung_nodule():
    from .datasets_lung import LungBothNodule
    luna_dir = "./LUNA16/cls/crop_v6"
    luna_label_path = "./LUNA16/cls/annotationdetclsconvfnl_v3.csv"
    meth_dir = "./Methodist_incidental/Nodules/patch32"
    meth_label_path = "./Methodist_incidental/Nodules/patch32/cls_labels.csv"
    dataset = LungBothNodule(luna_dir, luna_label_path, meth_dir, meth_label_path)
    return dataset

def build_dataset(is_train, transform=None, args=None):

    if transform is None:
        transform = build_transform(is_train, args)

    if args.datasource == "imagenet":
        dataset = init_imagenet(is_train, args.data_path, transform)
    elif args.datasource == "imagenet_limit":
        dataset = init_imagenet_limit(is_train, args.data_path, transform, args.num_tr, args.num_val, args.nb_classes)
    elif args.datasource == "imagenet_limit_with_name":
        dataset = init_imagenet_limit_with_name(is_train, args.data_path, transform, args.num_tr, args.num_val, args.nb_classes)
    elif args.datasource == "imagenet_limit_use_features":
        dataset = init_imagenet_limit_use_features(is_train, args.data_path, None)
    elif args.datasource == "lung":
        dataset = init_lung()
    elif args.datasource == "lung_nodule":
        dataset = init_lung_nodule()
    elif args.datasource == "luna_nodule":
        dataset = init_luna_nodule(is_train)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__ == '__main__':
    from main_finetune import get_args_parser
    args = get_args_parser()
    args = args.parse_args()
    is_train = True
    dataset = build_dataset(is_train, args)
    print("")