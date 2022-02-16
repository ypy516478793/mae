from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import glob, os

class Luna(Dataset):
    train_subsets = np.array(["subset0", "subset1", "subset2", "subset3",
                              "subset4", "subset5", "subset6", "subset7"])
    val_subsets = np.array(["subset8"])
    test_subsets = np.array(["subsets9"])

    def __init__(
            self,
            root_dir: str,
            split: str = "train",
            rand_rate: float = 0.5,
            crop_size: int = 32,
            pad_value: int = 170,
            order: str = "zyx",
            transform: Optional[Callable] = None,
        ):
        self.root_dir = root_dir
        self.transform = transform
        self.rand_rate = rand_rate
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.order = order

        self.images_list = []
        self.labels = []

        self.split = self._load_split(split)
        for s in self.split:
            self.file_list = glob.glob(os.path.join(self.root_dir, s, "*_clean.npy"))
            for img_path in self.file_list:
                dirname = os.path.dirname(img_path)
                filename = img_path.split("/")[-1].rstrip("_clean.npy")
                label = np.load(os.path.join(dirname, filename + '_label.npy'), allow_pickle=True)
                if np.all(label == 0):
                    label = np.array([])
                    self.labels.append(label)
                    self.images_list.append(img_path)
                else:
                    for l in label:  # Treat each nodule as an individual sample
                        self.labels.append(l)
                        self.images_list.append(img_path)

    def _load_split(self, split):
        if split == "trainVal":
            subsets = np.concatenate((Luna.train_subsets, Luna.val_subsets))
        elif split == "train":
            subsets = Luna.train_subsets
        elif split == "val":
            subsets = Luna.val_subsets
        else:
            subsets = Luna.test_subsets
            assert split == "test"
        return subsets

    def _image_crop(self, image, label):
        size = self.crop_size
        z, y, x = image.shape 
        if len(label) == 0:
            rand_crop = True
        else:
            rand_crop = np.random.random() < self.rand_rate
        if rand_crop:
            # randomly crop a region
            assert x >= size and y >= size and z >= size, "image shape is {:}".format(image.shape)
            zs = np.random.randint(0, z - size)
            ys = np.random.randint(0, y - size)
            xs = np.random.randint(0, x - size)
            cube = image[zs: zs + size,
                   ys: ys + size,
                   xs: xs + size]
            label = np.array([])
        else:
            # crop at the nodule
            image = np.pad(image, ((size, size), (size, size), (size, size)),
                           "constant", constant_values=self.pad_value)
            z, y, x = label[:3].astype(np.int) + size
            d = label[-1]
            cube = image[z - size // 2: z + (size + 1) // 2,
                   y - size // 2: y + (size + 1) // 2,
                   x - size // 2: x + (size + 1) // 2]
            label = np.array([size // 2, size // 2, size // 2, d])
        return cube, label


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = np.load(img_path, allow_pickle=True)
        image = image[0]
        if self.order == "xyz": # image shape == (32, 32, 32)
            image = image.transpose((2, 1, 0))
        label = self.labels[idx]
        img_crop, label = self._image_crop(image, label)
        img_crop = (img_crop / 255.).astype(np.float16)
        img_crop = np.expand_dims(img_crop, 0)

        if self.transform is not None:
            img_crop = self.transform(img_crop)

        return img_crop, label

class Luna_nodule(Dataset):
    classes = np.array(["Benign", "Malignant"])
    def __init__(
            self,
            root_dir: str,
            label_path: str,
            split: str = "train",
            order: str = "zyx",
            transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.label_path = label_path
        self.order = order
        self.transform = transform

        self.images_list = []
        self.labels = []

        label_df = pd.read_csv(self.label_path)
        alllst = label_df['seriesuid'].tolist()
        labellst = label_df['malignant'].tolist()  # 0: Benign, 1: Malignant
        for srsid, label in zip(alllst, labellst):
            image_path = os.path.join(self.root_dir, srsid + '.npy')
            self.images_list.append(image_path)
            self.labels.append(int(label))

        self.images_list, self.labels = self._load_split(split)

    def _load_split(self, split, rand_seed=42):
        image_trainVal, image_test, label_trainVal, label_test = train_test_split(
            self.images_list, self.labels, test_size=0.2, random_state=rand_seed)
        image_train, image_val, label_train, label_val = train_test_split(
            image_trainVal, label_trainVal, test_size=0.2, random_state=rand_seed)
        if split == "trainVal":
            return image_trainVal, label_trainVal
        elif split == "train":
            return image_train, label_train
        elif split == "val":
            return image_val, label_val
        else:
            assert split == "test"
            return image_test, label_test

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = np.load(img_path, allow_pickle=True) # image shape == (32, 32, 32)
        if self.order == "xyz": # image shape == (32, 32, 32)
            image = image.transpose((2, 1, 0))
        label = self.labels[idx]
        image = (image / 255.).astype(np.float16)
        image = np.expand_dims(image, 0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class Methodist(Dataset):
    test_fnames = ['patient002_20100910', 'patient002_20110314', 'patient002_20120906', 'patient002_20090310',
                   'patient005_20120524', 'patient006_20121023', 'patient011_20120626', 'patient011_20121015',
                   'patient012_20121204', 'patient016_20121127', 'patient017_20130102', 'patient018_20121001',
                   'patient018_20130110', 'patient021_20120508', 'patient021_20121113', 'patient021_20130212',
                   'patient023_20130316', 'patient025_20130226', 'patient032_20130509', 'patient033_20130514',
                   'patient034_20121002', 'patient034_20121228', 'patient034_20130423', 'patient035_20160830',
                   'patient037_20110516', 'patient037_20111028', 'patient037_20130502', 'patient037_20130510',
                   'patient040_20130722', 'patient041_20130319', 'patient041_20130614', 'patient041_20130723',
                   'patient045_20130718', 'patient046_20130603', 'patient046_20130826', 'patient047_20130821',
                   'patient049_20130820', 'patient050_20130820', 'patient051_20130905', 'patient052_20130516',
                   'patient053_20130912', 'patient055_20130925']
    max_nodule_size = 60
    def __init__(
            self,
            root_dir: str,
            split: str = "train",
            rand_rate: float = 0.5,
            crop_size: int = 32,
            pad_value: int = 170,
            order: str = "zyx",
            transform: Optional[Callable] = None,
        ):
        self.root_dir = root_dir
        self.transform = transform
        self.rand_rate = rand_rate
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.order = order

        self.images_list = []
        self.labels = []

        self.file_list = glob.glob(self.root_dir + "/*/*_clean.npz")
        self.file_list = self._load_split(split)

        for img_path in self.file_list:
            dirname = os.path.dirname(img_path)
            filename = img_path.split("/")[-1].rstrip("_clean.npz")
            label = np.load(os.path.join(dirname, filename + '_label.npz'), allow_pickle=True)["label"]
            label = label[label[:, -1] < Methodist.max_nodule_size]  ## remove nodule larger than specific size
            if np.all(label == 0):
                label = np.array([])
                self.labels.append(label)
                self.images_list.append(img_path)
            else:
                for l in label:  # Treat each nodule as an individual sample
                    self.labels.append(l)
                    self.images_list.append(img_path)

    def _load_split(self, split, rand_seed=42, fixTest=True):
        if fixTest:
            image_trainVal, image_test = [], []
            for img_path in self.file_list:
                fname = img_path.split("/")[-1].rstrip("_clean.npz")
                if fname in Methodist.test_fnames:
                    image_test.append(img_path)
                else:
                    image_trainVal.append(img_path)
            image_train, image_val = train_test_split(
                image_trainVal, test_size=0.2, random_state=rand_seed)
        else:
            image_trainVal, image_test = train_test_split(
                self.file_list, test_size=0.2, random_state=rand_seed)
            image_train, image_val = train_test_split(
                image_trainVal, test_size=0.2, random_state=rand_seed)
        if split == "trainVal":
            return image_trainVal
        elif split == "train":
            return image_train
        elif split == "val":
            return image_val
        else:
            assert split == "test"
            return image_test

    def _image_crop(self, image, label):
        size = self.crop_size
        z, y, x = image.shape
        if len(label) == 0:
            rand_crop = True
        else:
            rand_crop = np.random.random() < self.rand_rate
        if rand_crop:
            # randomly crop a region
            assert x >= size and y >= size and z >= size, "image shape is {:}".format(image.shape)
            zs = np.random.randint(0, z - size)
            ys = np.random.randint(0, y - size)
            xs = np.random.randint(0, x - size)
            cube = image[zs: zs + size,
                   ys: ys + size,
                   xs: xs + size]
            label = np.array([])
        else:
            # crop at the nodule
            image = np.pad(image, ((size, size), (size, size), (size, size)),
                           "constant", constant_values=self.pad_value)
            z, y, x = label[:3].astype(np.int) + size
            d = label[-1]
            cube = image[z - size // 2: z + (size + 1) // 2,
                   y - size // 2: y + (size + 1) // 2,
                   x - size // 2: x + (size + 1) // 2]
            label = np.array([size // 2, size // 2, size // 2, d])
        return cube, label

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = np.load(img_path, allow_pickle=True)["image"]
        image = image[0]
        if self.order == "xyz": # image shape == (32, 32, 32)
            image = image.transpose((2, 1, 0))
        label = self.labels[idx]
        img_crop, label = self._image_crop(image, label)
        img_crop = (img_crop / 255.).astype(np.float16)
        img_crop = np.expand_dims(img_crop, 0)

        if self.transform is not None:
            img_crop = self.transform(img_crop)

        return img_crop, label


class Methodist_nodule(Dataset):
    classes = np.array(["Benign", "Malignant"])

    def __init__(
            self,
            root_dir: str,
            label_path: str,
            split: str = "train",
            order: str = "zyx",
            transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.label_path = label_path
        self.order = order
        self.transform = transform

        self.images_list = []
        self.labels = []

        label_df = pd.read_csv(self.label_path)
        alllst = label_df['noduleId'].tolist()
        labellst = label_df['malignant'].tolist()  # 0: Benign, 1: Malignant
        for srsid, label in zip(alllst, labellst):
            image_path = os.path.join(self.root_dir, srsid + '.npy')
            self.images_list.append(image_path)
            self.labels.append(int(label))

        self.images_list, self.labels = self._load_split(split)

    def _load_split(self, split, rand_seed=42):
        image_trainVal, image_test, label_trainVal, label_test = train_test_split(
            self.images_list, self.labels, test_size=0.2, random_state=rand_seed)
        image_train, image_val, label_train, label_val = train_test_split(
            image_trainVal, label_trainVal, test_size=0.2, random_state=rand_seed)
        if split == "trainVal":
            return image_trainVal, label_trainVal
        elif split == "train":
            return image_train, label_train
        elif split == "val":
            return image_val, label_val
        else:
            assert split == "test"
            return image_test, label_test

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = np.load(img_path, allow_pickle=True)  # image shape == (32, 32, 32)
        if self.order == "xyz":  # image shape == (32, 32, 32)
            image = image.transpose((2, 1, 0))
        label = self.labels[idx]
        image = (image / 255.).astype(np.float16)
        image = np.expand_dims(image, 0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class LungBothNodule(Dataset):
    max_nodule_size = 60
    def __init__(
            self,
            luna_dir: str,
            luna_label_path: str,
            meth_dir: str,
            meth_label_path: str,
            order: str = "zyx",
            transform: Optional[Callable] = None,
        ):
        self.luna_dir = luna_dir
        self.luna_label_path = luna_label_path
        self.meth_dir = meth_dir
        self.meth_label_path = meth_label_path
        self.transform = transform
        self.order = order

        self.images_list = []
        self.labels = []

        # Load luna data
        label_df = pd.read_csv(self.luna_label_path)
        alllst = label_df['seriesuid'].tolist()
        labellst = label_df['malignant'].tolist()  # 0: Benign, 1: Malignant
        for srsid, label in zip(alllst, labellst):
            image_path = os.path.join(self.luna_dir, srsid + '.npy')
            self.images_list.append(image_path)
            self.labels.append(int(label))

        # Load methodist data
        label_df = pd.read_csv(self.meth_label_path)
        alllst = label_df['noduleId'].tolist()
        labellst = label_df['malignant'].tolist()  # 0: Benign, 1: Malignant
        for srsid, label in zip(alllst, labellst):
            image_path = os.path.join(self.meth_dir, srsid + '.npy')
            self.images_list.append(image_path)
            self.labels.append(int(label))

        # Shuffle
        xy = list(zip(self.images_list, self.labels))
        np.random.shuffle(xy)
        self.images_list, self.labels = zip(*xy)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = np.load(img_path, allow_pickle=True)  # image shape == (32, 32, 32)
        if self.order == "xyz":  # image shape == (32, 32, 32)
            image = image.transpose((2, 1, 0))
        label = self.labels[idx]
        image = (image / 255.).astype(np.float16)
        image = np.expand_dims(image, 0)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class LungBoth(Dataset):
    max_nodule_size = 60
    def __init__(
            self,
            luna_dir: str,
            meth_dir: str,
            rand_rate: float = 0.5,
            crop_size: int = 32,
            pad_value: int = 170,
            order: str = "zyx",
            transform: Optional[Callable] = None,
        ):
        self.luna_dir = luna_dir
        self.meth_dir = meth_dir
        self.transform = transform
        self.rand_rate = rand_rate
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.order = order

        self.images_list = []
        self.labels = []

        # Load luna data
        self.file_list = glob.glob(os.path.join(self.luna_dir, "*/*_clean.npy"))
        for img_path in self.file_list:
            dirname = os.path.dirname(img_path)
            filename = img_path.split("/")[-1].rstrip("_clean.npy")
            label = np.load(os.path.join(dirname, filename + '_label.npy'), allow_pickle=True)
            label = label[label[:, -1] < LungBoth.max_nodule_size]  ## remove nodule larger than specific size
            if np.all(label == 0):
                label = np.array([])
                self.labels.append(label)
                self.images_list.append(img_path)
            else:
                for l in label:  # Treat each nodule as an individual sample
                    self.labels.append(l)
                    self.images_list.append(img_path)

        # Load methodist data
        self.file_list = glob.glob(os.path.join(self.meth_dir, "*/*_clean.npz"))
        for img_path in self.file_list:
            dirname = os.path.dirname(img_path)
            filename = img_path.split("/")[-1].rstrip("_clean.npz")
            label = np.load(os.path.join(dirname, filename + '_label.npz'), allow_pickle=True)["label"]
            label = label[label[:, -1] < LungBoth.max_nodule_size]  ## remove nodule larger than specific size
            if np.all(label == 0):
                label = np.array([])
                self.labels.append(label)
                self.images_list.append(img_path)
            else:
                for l in label:  # Treat each nodule as an individual sample
                    self.labels.append(l)
                    self.images_list.append(img_path)

        # Shuffle
        xy = list(zip(self.images_list, self.labels))
        np.random.shuffle(xy)
        self.images_list, self.labels = zip(*xy)


    def _image_crop(self, image, label):
        size = self.crop_size
        z, y, x = image.shape
        if len(label) == 0:
            rand_crop = True
        else:
            rand_crop = np.random.random() < self.rand_rate
        if rand_crop:
            # randomly crop a region
            assert x >= size and y >= size and z >= size, "image shape is {:}".format(image.shape)
            zs = np.random.randint(0, z - size)
            ys = np.random.randint(0, y - size)
            xs = np.random.randint(0, x - size)
            cube = image[zs: zs + size,
                   ys: ys + size,
                   xs: xs + size]
            label = np.array([0, 0, 0, 0])
        else:
            # crop at the nodule
            image = np.pad(image, ((size, size), (size, size), (size, size)),
                           "constant", constant_values=self.pad_value)
            z, y, x = label[:3].astype(np.int) + size
            d = label[-1]
            cube = image[z - size // 2: z + (size + 1) // 2,
                   y - size // 2: y + (size + 1) // 2,
                   x - size // 2: x + (size + 1) // 2]
            label = np.array([size // 2, size // 2, size // 2, d])
        return cube, label

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = np.load(img_path, allow_pickle=True)
        if img_path.endswith(".npz"):
            image = image["image"]
        image = image[0]
        if self.order == "xyz": # image shape == (32, 32, 32)
            image = image.transpose((2, 1, 0))
        label = self.labels[idx]
        img_crop, label = self._image_crop(image, label)
        img_crop = (img_crop / 255.).astype(np.float16)
        img_crop = np.expand_dims(img_crop, 0)

        if self.transform is not None:
            img_crop = self.transform(img_crop)

        return img_crop, 0