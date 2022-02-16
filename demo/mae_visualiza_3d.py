import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from skimage.util import montage
import models_mae_3d as models_mae
# import models_mae

# define the utils

# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])

imagenet_mean = np.average(np.array([0.485, 0.456, 0.406]))
imagenet_std = np.average(np.array([0.229, 0.224, 0.225]))


def show_image(image, title=''):
    # image is [H, W, 3]
    if image.shape[2] == 3:
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
    else:
        plt.imshow(np.clip((montage(image) * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.int), cmap="gray")
        plt.title(title, fontsize=16)
        plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)(img_size=32, in_chans=1)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, model, mr=0.75):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0) # num_channel = 1

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=mr)
    print("loss is: ", loss)
    y = model.unpatchify(y)
    # y = torch.einsum('ncxyz->nxyzc', y).detach().cpu()
    y = y[:, 0].detach().cpu()  # shape == (1, 1, 32, 32, 32)

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3 * 1)  # (N, H*W, p*p*3) -> (N, Z*Y*X, p*p*p*1)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = mask[:, 0].detach().cpu()
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # x = torch.einsum('nchw->nhwc', x)
    x = x[:, 0].detach().cpu()

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.figure(figsize=(16, 5))
    # plt.rcParams['figure.figsize'] = [48, 48]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.tight_layout()
    plt.show()

# load an image
img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
# img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851

# scan_path = "../LUNA16/cls/crop_v5/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860-111.npy"
# scan_path = "../LUNA16/cls/crop_v6/1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860-111.npy"
# scan_path = "../LUNA16/cls/crop_v6/1.3.6.1.4.1.14519.5.2.1.6279.6001.108231420525711026834210228428-951.npy"
scan_path = "../LUNA16/cls/crop_v6/1.3.6.1.4.1.14519.5.2.1.6279.6001.100953483028192176989979435275-171.npy"
img = np.load(scan_path)
img = np.array(img) / 255.

assert img.shape == (32, 32, 32)

# img = Image.open(requests.get(img_url, stream=True).raw)
# img = img.resize((224, 224))
# img = np.array(img) / 255.
#
# assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
img = img - imagenet_mean
img = img / imagenet_std

# plt.rcParams['figure.figsize'] = [5, 5]
# show_image(torch.tensor(img))

# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)


mr = 0.75
# chkpt_dir = '../mae_pretrain_vit_large.pth'
# chkpt_dir = '../jobdir/pretrain_lung_single/vit_large_patch16_e800_ft100_blr5e2_wu40/checkpoint-99.pth'
# chkpt_dir = '../jobdir/pretrain_lung/vit_large_patch4_e800_input32_luna_mr75/checkpoint-799.pth'
# chkpt_dir = '../jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung/checkpoint-799.pth'
# chkpt_dir = '../jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr5/checkpoint-600.pth'
# chkpt_dir = '../jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr6/checkpoint-600.pth'
# chkpt_dir = '../jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr7/checkpoint-600.pth'
# chkpt_dir = '../jobdir/pretrain_lung/vit_large_patch16_e800_crop32_lung_mr8/checkpoint-700.pth'
# chkpt_dir = '../jobdir/pretrain_lung_nodule/vit_large_patch8_e800_crop32_lung_mr75/checkpoint-600.pth'
# chkpt_dir = '../jobdir/pretrain_lung_nodule/vit_large_patch8_e2400_crop32_lung_mr75/checkpoint-2399.pth'
# chkpt_dir = '../jobdir/pretrain_lung_nodule/vit_large_patch8_e3600_crop32_lung_mr75_blr1.5e4_wu1000/checkpoint-3599.pth'
chkpt_dir = '../jobdir/pretrain_lung_nodule/vit_large_patch8_e3600_crop32_lung_mr75_blr1.5e4_wu1000_debug/checkpoint-10.pth'

model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
# model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch4')
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
# torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae, mr=mr)