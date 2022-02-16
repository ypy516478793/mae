from torch.utils.data import Dataset

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torch
import os



class Features(Dataset):

    def __init__(
            self,
            root_dir: str,
            transform: Optional[Callable] = None,
        ):
        self.root_dir = root_dir
        self.transform = transform
        classes, class_to_idx = self._find_classes(self.root_dir)
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.images_list = []
        self.labels = []

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(self.root_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if path.endswith("pt"):
                        self.images_list.append(path)
                        self.labels.append(class_index)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images_list[idx]
        image = torch.load(img_path)["feature"]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

