from random import randrange
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

import numpy as np
from shutil import rmtree

from nuwa_pytorch.tokenizer import tokenizer
from nuwa_pytorch.optimizer import get_optimizer
from nuwa_pytorch.image_utils import gif_to_tensor
from nuwa_pytorch import NUWA

import torchvision.transforms as T
from torchvision.utils import make_grid, save_image

# helper functions

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# dataloader helper functions

def pad_collate_fn(batch):
    texts, videos = zip(*batch)
    return pad_sequence(texts, batch_first = True), torch.stack(videos)

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
        random_rotate = False
    ):
        super().__init__()
        self.num_videos = num_videos
        self.videos_memmap = np.memmap(videos_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_frames, channels, image_size, image_size))
        self.labels_memmap = np.memmap(labels_memmap_path, mode = 'r', dtype = np.uint8, shape = (num_videos, num_digits))
        self.random_rotate = random_rotate

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        video = torch.from_numpy(self.videos_memmap[idx].copy()).float()
        label = torch.from_numpy(self.labels_memmap[idx].copy())

        video /= 255
        video = video.to(torch.float32)

        text = tokenizer.encode(' '.join(map(str, label.tolist())))
        text = torch.Tensor(text).long()

        if self.random_rotate:
            video = T.functional.rotate(video, choice([0, 90, 180, 270]))

        return text, video

# dataset for training from folder of videos as gifs

class GifVideoDataset(Dataset):
    def __init__(
        self,
        *,
        folder,
        channels = 1
    ):
        folder = Path(folder)
        gifs = folder.glob('**/*.gif')
        txts = folder.glob('**/*.txt')

        gif_path_stems = set(map(lambda t: str(t.with_suffix('')), gifs))
        txt_path_stems = set(map(lambda t: str(t.with_suffix('')), txts))
        self.path_stems = list(gif_path_stems.intersection(txt_path_stems))

        self.channels = channels
        print(f'{len(self.path_stems)} video / text pairs found')

    def __len__(self):
        return len(self.path_stems)

    def __getitem__(self, idx):
        path_stem = self.path_stems[idx]

        txt_path = Path(f'{path_stem}.txt')
        txt_str = txt_path.read_text()
        text_tensor = torch.Tensor(tokenizer.encode(txt_str)).long()

        video_tensor = gif_to_tensor(f'{path_stem}.gif', channels = self.channels)
        return text_tensor, video_tensor

# training class

class NUWATrainer(nn.Module):
    def __init__(
        self,
        *,
        nuwa,
        dataset,
        num_train_steps,
        lr = 3e-4,
        wd = 0.01,
        batch_size = 4,
        grad_accum_every = 8,
        max_grad_norm = 0.5,
        save_model_every = 2500,
        save_results_every = 1000,
        results_folder = './results-nuwa',
        num_sampled_frames = float('inf')
    ):
        super().__init__()
        assert isinstance(nuwa, NUWA), 'nuwa must be an instance of NUWA'
        self.nuwa = nuwa

        self.steps = 0
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm

        self.optim = get_optimizer(nuwa.parameters(), lr = lr, wd = wd)

        # dataset

        self.ds = dataset

        # dataloader

        self.dl = cycle(DataLoader(
            self.ds,
            batch_size = batch_size,
            collate_fn = pad_collate_fn,
            shuffle = True
        ))

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.num_sampled_frames = num_sampled_frames

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def train_step(self):
        device = next(self.nuwa.parameters()).device
        self.nuwa.train()

        logs = {}

        for _ in range(self.grad_accum_every):
            text, video = next(self.dl)
            text, video = map(lambda t: t.to(device), (text, video))

            loss = self.nuwa(
                text = text,
                video = video,
                return_loss = True
            )
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

            (loss / self.grad_accum_every).backward()

        print(f'{self.steps} loss: {logs["loss"]}')

        torch.nn.utils.clip_grad_norm_(self.nuwa.parameters(), self.max_grad_norm)
        self.optim.step()
        self.optim.zero_grad()

        if not (self.steps % self.save_results_every):
            self.nuwa.eval()
            print(f'{self.steps} sampling')

            rand_idx = randrange(0, len(self.ds))

            text, video = self.ds[rand_idx]
            text, video = next(self.dl)
            text = text.to(device)

            video = self.nuwa.generate(text = text, num_frames = min(video.shape[1], self.num_sampled_frames))
            one_video = video[0].cpu().clamp(0., 1.)

            text_str = tokenizer.decode(text[0])

            logs['sampled_text'] = text_str
            logs['sampled_video'] = one_video.numpy()

            image = rearrange(one_video, 'f c h w -> c (f h) w')
            save_image(image, str(self.results_folder / f'{self.steps}.png'))

            print(f'{self.steps}: saving to {str(self.results_folder)}')

        if not (self.steps % self.save_model_every):
            state_dict = self.nuwa.state_dict()
            model_path = str(self.results_folder / f'nuwa.{self.steps}.pt')
            torch.save(state_dict, model_path)

            print(f'{self.steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        print('training complete')
