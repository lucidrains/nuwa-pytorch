import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
from nuwa_pytorch.tokenizer import tokenizer
from nuwa_pytorch.optimizer import get_optimizer

# helper functions

def exists(val):
    return val is not None

# dataset class

class MnistDataset(Dataset):
    def __init__(
        self,
        num_videos,
        videos_memmap_path,
        labels_memmap_path,
        num_digits = 2,
        num_frames = 10,
        image_size = 64,
        channels = 1,
    ):
        super().__init__()
        self.num_videos = num_videos
        self.videos_memmap = np.memmap(videos_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_frames, channels, image_size, image_size))
        self.labels_memmap = np.memmap(labels_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_digits))

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        video = torch.from_numpy(self.videos_memmap[idx].copy()).float()
        label = torch.from_numpy(self.labels_memmap[idx].copy())

        video /= 255
        video = video.to(torch.float32)

        text = tokenizer.encode(' '.join(map(str, label.tolist())))
        text = torch.Tensor(text).long()

        return text, video

# training class
