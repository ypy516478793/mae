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

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from copy import deepcopy
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory
import json, os

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

ssl_model_dict = {}
ssl_model_test_dict = {}

def train_one_epoch_meta(model: torch.nn.Module, model_cls: torch.nn.Module,
                         criterion: torch.nn.Module, data_loader: Iterable,
                         optimizer: torch.optim.Optimizer,
                         optimizer_cls: torch.optim.Optimizer,
                         device: torch.device, epoch: int,
                         loss_scaler, loss_scaler_cls,
                         max_norm: float = None,
                         mixup_fn: Optional[Mixup] = None,
                         log_writer=None,
                         args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Meta Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if epoch == 0:
            model_ssl = deepcopy(model)
            model_ssl_without_ddp = model_ssl.module
            param_groups_ssl = optim_factory.add_weight_decay(model_ssl_without_ddp, args.weight_decay)
            optimizer_ssl = torch.optim.AdamW(param_groups_ssl, lr=args.lr_ssl, betas=(0.9, 0.95))
            optimizer_ssl.load_state_dict(optimizer.state_dict())
            loss_scaler_ssl = deepcopy(loss_scaler)

            for epoch_ssl in range(0, args.epochs_ssl):
                # if args.distributed:
                #     data_loader_train.sampler.set_epoch(epoch)

                data_loader_train_ssl = simple_iterator((samples,))
                train_stats = train_one_epoch(
                    model_ssl, data_loader_train_ssl,
                    optimizer_ssl, device, epoch_ssl, loss_scaler_ssl,
                    log_writer=log_writer,
                    args=args
                )
                # if args.output_dir and (epoch_ssl + 1 == args.epochs_ssl):
                #     misc.save_model_ssl(
                #         args=args, model=model_ssl, model_without_ddp=model_ssl_without_ddp, optimizer=optimizer_ssl,
                #         loss_scaler=loss_scaler_ssl, epoch=epoch_ssl, idx=data_iter_step)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'meta_epoch': epoch, 'ssl_epoch': epoch_ssl,}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log_{:d}.txt".format(data_iter_step)), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
            ssl_model_dict[data_iter_step] = model_ssl_without_ddp.cpu()
            # model.module.load_state_dict(model_ssl_without_ddp.state_dict())
            del model_ssl
            del optimizer_ssl
            del loss_scaler_ssl
        else:
            model_ssl_without_ddp = ssl_model_dict[data_iter_step]

        # misc.initial_model_cls(args, model_cls.module, model_ssl_without_ddp)
        model_cls_without_ddp = model_cls.module
        misc.initial_model_cls(args, model_cls_without_ddp, model_ssl_without_ddp)

        del model_ssl_without_ddp

        # # freeze all but the head
        # for _, p in model_cls_without_ddp.named_parameters():
        #     p.requires_grad = False
        # for _, p in model_cls_without_ddp.head.named_parameters():
        #     p.requires_grad = True

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer_cls, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model_cls(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler_cls(loss, optimizer_cls, clip_grad=max_norm,
                    parameters=model_cls.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_cls.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer_cls.param_groups:
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


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    is_train: bool=True, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = ' ---------- Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_ssl(optimizer, data_iter_step / len(data_loader) + epoch, args)
        # for i in range(len(samples)):
        #
        #     model_learner = deepcopy(model)
        #     optimizer_learner = deepcopy(optimizer)
        #     loss_scaler_learner = NativeScaler()
        #     sample = samples[[i]].to(device, non_blocking=True)
        #
        #     epoch_learner = 100
        #     for e in range(epoch_learner):
        #
        #         if e % accum_iter == 0:
        #             lr_sched.adjust_learning_rate(optimizer_learner, e, args)
        #
        #         with torch.cuda.amp.autocast():
        #             loss, _, _ = model_learner(sample, mask_ratio=args.mask_ratio)
        #
        #         loss_value = loss.item()
        #
        #         if not math.isfinite(loss_value):
        #             print("Loss is {}, stopping training".format(loss_value))
        #             sys.exit(1)
        #
        #         loss /= accum_iter
        #         loss_scaler_learner(loss, optimizer_learner, parameters=model_learner.parameters(),
        #                             update_grad=(e + 1) % accum_iter == 0)
        #         if (e + 1) % accum_iter == 0:
        #             optimizer_learner.zero_grad()
        #
        #         if args.output_dir and (e % 100 == 0 or e + 1 == epoch_learner):
        #             misc.save_model(
        #                 args=args, model=model_learner, model_without_ddp=model_learner.module, optimizer=optimizer_learner,
        #                 loss_scaler=loss_scaler_learner, epoch=e)
        #

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if is_train:
                log_writer.add_scalar('train_loss_ssl', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('train_lr_ssl', lr, epoch_1000x)
            else:
                log_writer.add_scalar('test_loss_ssl', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('test_lr_ssl', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_cls(model_cls, criterion, images, target, metric_logger):
    # switch to evaluation mode
    model_cls.eval()

    # compute output
    with torch.cuda.amp.autocast():
        output = model_cls(images)
        loss = criterion(output, target)

    # acc1, acc5 = accuracy(output, target, topk=(1, 5))
    acc1 = accuracy(output, target, topk=(1,))[0]
    acc5 = torch.tensor(100)

    batch_size = images.shape[0]
    metric_logger.update(loss=loss.item())
    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

 # TODO: SSL ADAPTATION DURING THE TEST
def evaluate(data_loader: Iterable, model: torch.nn.Module, model_cls: torch.nn.Module,
             optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
             loss_scaler, log_writer=None, args=None):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.train(True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if epoch == 0:
            model_ssl = deepcopy(model)
            model_ssl_without_ddp = model_ssl.module
            param_groups_ssl = optim_factory.add_weight_decay(model_ssl_without_ddp, args.weight_decay)
            optimizer_ssl = torch.optim.AdamW(param_groups_ssl, lr=args.lr_ssl, betas=(0.9, 0.95))
            optimizer_ssl.load_state_dict(optimizer.state_dict())
            loss_scaler_ssl = deepcopy(loss_scaler)

            for epoch_ssl in range(0, args.epochs_ssl):
                # if args.distributed:
                #     data_loader_train.sampler.set_epoch(epoch)

                data_loader_train_ssl = simple_iterator((images,))
                train_stats = train_one_epoch(
                    model_ssl, data_loader_train_ssl,
                    optimizer_ssl, device, epoch_ssl, loss_scaler_ssl,
                    is_train=False,
                    log_writer=log_writer,
                    args=args
                )
                # if args.output_dir and (epoch_ssl + 1 == args.epochs_ssl):
                #     misc.save_model_ssl(
                #         args=args, model=model_ssl, model_without_ddp=model_ssl_without_ddp, optimizer=optimizer_ssl,
                #         loss_scaler=loss_scaler_ssl, epoch=epoch_ssl, idx=data_iter_step)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'ssl_epoch': epoch_ssl, }

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "test_log_{:d}.txt".format(data_iter_step)), mode="a",
                              encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            ssl_model_test_dict[data_iter_step] = model_ssl_without_ddp.cpu()
            # model.module.load_state_dict(model_ssl_without_ddp.state_dict())
            model_cls_without_ddp = model_cls.module
            misc.initial_model_cls(args, model_cls_without_ddp, model_ssl_without_ddp)

            del model_ssl
            del optimizer_ssl
            del loss_scaler_ssl
        else:
            model_ssl_without_ddp = ssl_model_test_dict[data_iter_step]

        model_cls_without_ddp = model_cls.module
        misc.initial_model_cls(args, model_cls_without_ddp, model_ssl_without_ddp)

        del model_ssl_without_ddp

        evaluate_cls(model_cls, criterion, images, target, metric_logger)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_meta_with_features(model: torch.nn.Module, model_cls: torch.nn.Module,
                         criterion: torch.nn.Module, data_loader: Iterable,
                         optimizer: torch.optim.Optimizer,
                         optimizer_cls: torch.optim.Optimizer,
                         device: torch.device, epoch: int,
                         loss_scaler, loss_scaler_cls,
                         max_norm: float = None,
                         mixup_fn: Optional[Mixup] = None,
                         log_writer=None,
                         args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Meta Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (image_paths, samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # temp
        exist = False
        for idx in range(len(samples)):
            path = image_paths[idx]
            path = path.replace("imagenet", args.feature_dir).replace("JPEG", "pt")
            if os.path.exists(path):
                exist = True
                break
        if exist:
            metric_logger.update(loss=0)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer_cls.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            metric_logger.update(lr=max_lr)
            continue
            


        if epoch == 0:
            model_ssl = deepcopy(model)
            model_ssl_without_ddp = model_ssl.module
            param_groups_ssl = optim_factory.add_weight_decay(model_ssl_without_ddp, args.weight_decay)
            optimizer_ssl = torch.optim.AdamW(param_groups_ssl, lr=args.lr_ssl, betas=(0.9, 0.95))
            optimizer_ssl.load_state_dict(optimizer.state_dict())
            loss_scaler_ssl = deepcopy(loss_scaler)

            for epoch_ssl in range(0, args.epochs_ssl):
                # if args.distributed:
                #     data_loader_train.sampler.set_epoch(epoch)

                data_loader_train_ssl = simple_iterator((samples,))
                train_stats = train_one_epoch(
                    model_ssl, data_loader_train_ssl,
                    optimizer_ssl, device, epoch_ssl, loss_scaler_ssl,
                    log_writer=log_writer,
                    args=args
                )
                # if args.output_dir and (epoch_ssl + 1 == args.epochs_ssl):
                #     misc.save_model_ssl(
                #         args=args, model=model_ssl, model_without_ddp=model_ssl_without_ddp, optimizer=optimizer_ssl,
                #         loss_scaler=loss_scaler_ssl, epoch=epoch_ssl, idx=data_iter_step)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'meta_epoch': epoch, 'ssl_epoch': epoch_ssl,}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log_{:d}.txt".format(data_iter_step)), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
            # ssl_model_dict[data_iter_step] = model_ssl_without_ddp.cpu()
            # model.module.load_state_dict(model_ssl_without_ddp.state_dict())
            del model_ssl
            del optimizer_ssl
            del loss_scaler_ssl
        else:
            model_ssl_without_ddp = ssl_model_dict[data_iter_step]

        # misc.initial_model_cls(args, model_cls.module, model_ssl_without_ddp)
        model_cls_without_ddp = model_cls.module
        misc.initial_model_cls(args, model_cls_without_ddp, model_ssl_without_ddp)

        del model_ssl_without_ddp

        # # freeze all but the head
        # for _, p in model_cls_without_ddp.named_parameters():
        #     p.requires_grad = False
        # for _, p in model_cls_without_ddp.head.named_parameters():
        #     p.requires_grad = True

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer_cls, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            features, outputs = model_cls(samples)
            loss = criterion(outputs, targets)

        for idx in range(len(samples)):
            path, feature, label = image_paths[idx], features[idx], targets[idx]
            path = path.replace("imagenet", args.feature_dir).replace("JPEG", "pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"feature": feature.cpu(), "label": label.cpu()}, path)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        # loss_scaler_cls(loss, optimizer_cls, clip_grad=max_norm,
        #             parameters=model_cls.parameters(), create_graph=False,
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_cls.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer_cls.param_groups:
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


def evaluate_with_features(data_loader: Iterable, model: torch.nn.Module, model_cls: torch.nn.Module,
             optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
             loss_scaler, log_writer=None, args=None):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.train(True)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        image_paths = batch[0]
        images = batch[1]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # temp
        exist = False
        for idx in range(len(images)):
            path = image_paths[idx]
            path = path.replace("imagenet", args.feature_dir).replace("JPEG", "pt")
            if os.path.exists(path):
                exist = True
                break
        if exist:
            batch_size = images.shape[0]
            metric_logger.update(loss=0)
            metric_logger.meters['acc1'].update(0, n=batch_size)
            metric_logger.meters['acc5'].update(100, n=batch_size)
            continue

        if epoch == 0:
            model_ssl = deepcopy(model)
            model_ssl_without_ddp = model_ssl.module
            param_groups_ssl = optim_factory.add_weight_decay(model_ssl_without_ddp, args.weight_decay)
            optimizer_ssl = torch.optim.AdamW(param_groups_ssl, lr=args.lr_ssl, betas=(0.9, 0.95))
            optimizer_ssl.load_state_dict(optimizer.state_dict())
            loss_scaler_ssl = deepcopy(loss_scaler)

            for epoch_ssl in range(0, args.epochs_ssl):
                # if args.distributed:
                #     data_loader_train.sampler.set_epoch(epoch)

                data_loader_train_ssl = simple_iterator((images,))
                train_stats = train_one_epoch(
                    model_ssl, data_loader_train_ssl,
                    optimizer_ssl, device, epoch_ssl, loss_scaler_ssl,
                    is_train=False,
                    log_writer=log_writer,
                    args=args
                )
                # if args.output_dir and (epoch_ssl + 1 == args.epochs_ssl):
                #     misc.save_model_ssl(
                #         args=args, model=model_ssl, model_without_ddp=model_ssl_without_ddp, optimizer=optimizer_ssl,
                #         loss_scaler=loss_scaler_ssl, epoch=epoch_ssl, idx=data_iter_step)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'ssl_epoch': epoch_ssl, }

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "test_log_{:d}.txt".format(data_iter_step)), mode="a",
                              encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

            # ssl_model_test_dict[data_iter_step] = model_ssl_without_ddp.cpu()
            # model.module.load_state_dict(model_ssl_without_ddp.state_dict())
            model_cls_without_ddp = model_cls.module
            misc.initial_model_cls(args, model_cls_without_ddp, model_ssl_without_ddp)

            del model_ssl
            del optimizer_ssl
            del loss_scaler_ssl
        else:
            model_ssl_without_ddp = ssl_model_test_dict[data_iter_step]

        model_cls_without_ddp = model_cls.module
        misc.initial_model_cls(args, model_cls_without_ddp, model_ssl_without_ddp)

        del model_ssl_without_ddp

        # switch to evaluation mode
        model_cls.eval()

        # compute output
        with torch.cuda.amp.autocast():
            features, output = model_cls(images)
            loss = criterion(output, target)

        # save hidden features
        for idx in range(len(images)):
            path, feature, label = image_paths[idx], features[idx], target[idx]
            path = path.replace("train", "test")
            path = path.replace("imagenet", args.feature_dir).replace("JPEG", "pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"feature": feature.cpu(), "label": label.cpu()}, path)

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