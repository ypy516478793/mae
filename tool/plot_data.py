from skimage.util import montage

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_luna(num_to_show):
    root_dir = "/home/pyuan2/Projects2021/mae/LUNA16/cls/crop_v5"
    image_list = [os.path.join(root_dir, i) for i in os.listdir(root_dir)] # [N, (32, 32, 32)]

    image_array = np.stack([np.load(i)[16] for i in image_list[:num_to_show]])
    image_array = montage(image_array, padding_width=1)

    plt.imshow(image_array, cmap="gray")
    plt.title("Luna nodule samples")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show(dpi=200)

def plot_methodsit(num_to_show):
    data_path = "/home/pyuan2/Projects2021/mae/Methodist_incidental/data_Ben/preprocessed_data_v1/Methodist_3Dcubes_p32.npz"

    image_array = np.load(data_path)["x"][:num_to_show, 0, 16] # (N, 1, 32, 32, 32)
    image_array = montage(image_array, padding_width=1)

    plt.imshow(image_array, cmap="gray")
    plt.title("Methodist nodule samples")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.show(dpi=200)

plot_luna(num_to_show=200)
# plot_methodsit(num_to_show=200)