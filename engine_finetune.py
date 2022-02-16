# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import os

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1,))[0]
        acc5 = torch.tensor(100)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_with_features(data_loader, model, device, log_writer, feature_dir, is_train=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_features = []
    all_labels = []
    all_images = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image_paths = batch[0]
        images = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            features, output = model(images)
            loss = criterion(output, target)

        # embedding visualization
        if log_writer is not None:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images.cpu()
            images = torch.clip((images * imagenet_std + imagenet_mean) * 255, 0, 255) / 255
            all_features.append(features.cpu())
            all_labels.append(target.cpu())
            all_images.append(images)

        # # save hidden features
        # for idx in range(len(images)):
        #     path, feature, label = image_paths[idx], features[idx], target[idx]
        #     if not is_train:
        #         path = path.replace("train", "test")
        #     path = path.replace("imagenet", feature_dir).replace("JPEG", "pt")
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     torch.save({"feature": feature.cpu(), "label": label.cpu()}, path)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1,))[0]
        acc5 = torch.tensor(100)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # embedding visualization
    if log_writer is not None:
        log_writer.add_embedding(torch.cat(all_features),
                                 torch.cat(all_labels),
                                 torch.cat(all_images))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate_use_features(data_loader, model, device, log_writer, feature_dir, is_train=True):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    all_features = []
    all_labels = []
    all_images = []

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        image_paths = batch[0]
        images = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # # compute output
        # with torch.cuda.amp.autocast():
        #     features, output = model(images)
        #     loss = criterion(output, target)
        # compute output

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        features = images

        from PIL import Image
        file_list = [i.replace("imagenet_features/meta_b1", "imagenet").replace("pt", "JPEG")
                     for i in image_paths]
        def pil_loader(path):
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        img_list = [pil_loader(i) for i in file_list]
        import torchvision.transforms as transforms
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        sample_list = [transform_val(i) for i in img_list]
        images = torch.stack(sample_list).to(device)

        # embedding visualization
        if log_writer is not None:
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = images.cpu()
            images = torch.clip((images * imagenet_std + imagenet_mean) * 255, 0, 255) / 255
            all_features.append(features.cpu())
            all_labels.append(target.cpu())
            all_images.append(images)

        # # save hidden features
        # for idx in range(len(images)):
        #     path, feature, label = image_paths[idx], features[idx], target[idx]
        #     if not is_train:
        #         path = path.replace("train", "test")
        #     path = path.replace("imagenet", feature_dir).replace("JPEG", "pt")
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     torch.save({"feature": feature.cpu(), "label": label.cpu()}, path)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1,))[0]
        acc5 = torch.tensor(100)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # embedding visualization
    if log_writer is not None:
        log_writer.add_embedding(torch.cat(all_features),
                                 torch.cat(all_labels),
                                 torch.cat(all_images))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}