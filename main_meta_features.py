# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.pos_embed import interpolate_pos_embed

import models_mae
# import models_mae_3d as models_mae
# import models_vit
import models_vit_with_feature as models_vit

from engine_pretrain_meta import train_one_epoch_meta, evaluate
from copy import deepcopy

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--epochs_ssl', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0., metavar='PCT',
                        help='Drop path rate (default: 0.1 for finetuning)')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--lr_ssl', type=float, default=None, metavar='LR',
                        help='learning rate for ssl (absolute lr)')
    parser.add_argument('--blr_ssl', type=float, default=1e-3, metavar='LR',
                        help='base learning rate for ssl: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--warmup_epochs_ssl', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--feature_dir', default=None, type=str,
                        help='feature directory to save')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # My arguments
    parser.add_argument("-d", "--datasource",
                        help="Name of the available datasets",
                        choices=["imagenet", "imagenet_limit", "lung"])
    parser.add_argument('--num_tr', default=100, type=int)
    parser.add_argument('--num_val', default=300, type=int)

    return parser


def single_image():
    import requests
    from PIL import Image
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'  # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851

    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    img = img.astype(np.float16)

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    return x

def single_lung():
    scan_path = "LUNA16/cls/crop_v6/1.3.6.1.4.1.14519.5.2.1.6279.6001.108231420525711026834210228428-951.npy"
    # scan_path = "LUNA16/cls/crop_v6/1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275-171.npy"
    img = np.load(scan_path)
    img = np.array(img) / 255.

    imagenet_mean = np.average(np.array([0.485, 0.456, 0.406]))
    imagenet_std = np.average(np.array([0.229, 0.224, 0.225]))

    assert img.shape == (32, 32, 32)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std
    img = img.astype(np.float16)

    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)  # channel
    x = x.unsqueeze(dim=0)  # batch
    # x = torch.einsum('nhwc->nchw', x)

    return x

class simple_iterator:
    def __init__(self, img_list):
        self.img_list = img_list
        self.current = -1

    def __len__(self):
        return len(self.img_list)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.img_list):
            return self.img_list[self.current], None
        raise StopIteration

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_train = build_dataset(is_train=True, transform=transform_train, args=args)
    dataset_val = build_dataset(is_train=False, transform=transform_val, args=args)
    # print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.log_dir is None:
        args.log_dir = args.output_dir
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    ## define the model
    in_chans = 1 if args.datasource == "lung" else 3
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, img_size=args.input_size, in_chans=in_chans)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## define the cls model
    in_chans = 1 if "nodule" in args.datasource else 3
    model_cls_name = args.model.lstrip("mae_")
    model_cls = models_vit.__dict__[model_cls_name](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.input_size,
        in_chans=in_chans
    )

    # interpolate position embedding
    interpolate_pos_embed(model_cls, model.state_dict())

    # manually initialize fc layer
    trunc_normal_(model_cls.head.weight, std=2e-5)

    # for linear prob only
    # hack: revise model's head with BN
    model_cls.head = torch.nn.Sequential(torch.nn.BatchNorm1d(
        model_cls.head.in_features, affine=False, eps=1e-6), model_cls.head)

    # freeze all but the head
    for _, p in model_cls.named_parameters():
        p.requires_grad = False
    for _, p in model_cls.head.named_parameters():
        p.requires_grad = True

    model_cls.to(device)

    model_cls_without_ddp = model_cls
    print("Meta Model = %s" % str(model_cls_without_ddp))
    n_parameters_meta = sum(p.numel() for p in model_cls.parameters() if p.requires_grad)


    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    ## ssl learning rate
    if args.lr_ssl is None:  # only base_lr is specified
        args.lr_ssl = args.blr_ssl * eff_batch_size / 256

    print("base lr for ssl: %.2e" % (args.lr_ssl * 256 / eff_batch_size))
    print("actual lr for ssl: %.2e" % args.lr_ssl)

    ## cls learning rate
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model_cls = torch.nn.parallel.DistributedDataParallel(model_cls, device_ids=[args.gpu], find_unused_parameters=False)
        model_cls_without_ddp = model_cls.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr_ssl, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    optimizer_cls = LARS(model_cls_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer_cls)
    loss_scaler_cls = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    checkpoint = torch.load(args.finetune, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, start_over=True)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch_meta(
            model, model_cls,
            criterion, data_loader_train,
            optimizer, optimizer_cls,
            device, epoch,
            loss_scaler, loss_scaler_cls,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 100 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val, model, model_cls, optimizer, device, epoch,
                              loss_scaler, log_writer=log_writer, args=args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters,
                        'n_parameters_meta': n_parameters_meta
                     }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
