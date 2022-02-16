from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


class DeepFeatures(object):
    def __init__(self, imgs_folder, embs_folder, tb_folder):
        self.imgs_folder = imgs_folder
        self.embs_folder = embs_folder
        self.tb_folder = tb_folder

        self.writer = None
