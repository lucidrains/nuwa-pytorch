from math import sqrt
from random import choice
from pathlib import Path
from shutil import rmtree

import torch
from torch import nn
from torch.optim import AdamW, Adam
import numpy as np

from PIL import Image
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

from einops import rearrange
from nuwa_pytorch.nuwa_pytorch import VQGanVAE

# helpers

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

# adamw functions

def separate_weight_decayable_params(params):
    no_wd_params = set([param for param in params if param.ndim < 2])
    wd_params = set(params) - no_wd_params
    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 3e-4,
    wd = 1e-1,
    filter_by_requires_grad = False
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr = lr)

    params = set(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': list(wd_params)},
        {'params': list(no_wd_params), 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = wd)

# classes

class MemmappedImageDataset(Dataset):
    def __init__(
        self,
        *,
        path,
        shape,
        random_rotate = True
    ):
        super().__init__()
        path = Path(path)
        assert path.exists(), f'path {path} must exist'
        self.memmap = np.memmap(str(path), mode = 'r', dtype = np.uint8, shape = shape)
        self.random_rotate = random_rotate

        image_size = shape[-1]
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return self.memmap.shape[0]

    def __getitem__(self, index):
        arr = self.memmap[index]

        if arr.shape[0] == 1:
            arr = rearrange(arr, '1 ... -> ...')

        img = Image.fromarray(arr)
        img = self.transform(img)

        if self.random_rotate:
            img = T.functional.rotate(img, choice([0, 90, 180, 270]))
        return img

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        num_train_steps,
        lr,
        batch_size,
        grad_accum_every,
        wd = 0.,
        images_memmap_path = None,
        images_memmap_shape = None,
        folder = None,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results'
    ):
        super().__init__()
        assert isinstance(vae, VQGanVAE), 'vae must be instance of VQGanVAE'
        image_size = vae.image_size
        self.vae = vae

        self.steps = 0
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        self.optim = get_optimizer(vae.parameters(), lr = lr, wd = wd)

        # create dataset

        assert exists(folder) ^ exists(images_memmap_path), 'either folder or memmap path to images must be supplied'

        if exists(images_memmap_path):
            assert exists(images_memmap_shape), 'shape of memmapped images must be supplied'

        if exists(folder):
            self.ds = ImageDataset(folder, image_size = image_size)
        elif exists(images_memmap_path):
            self.ds = MemmappedImageDataset(path = images_memmap_path, shape = images_memmap_shape)

        # dataloader

        self.dl = cycle(DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        ))

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

    def train_step(self):
        device = next(self.vae.parameters()).device
        self.vae.train()

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.grad_accum_every):
            img = next(self.dl)
            img = img.to(device)

            loss = self.vae(img, return_loss = True)
            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

            (loss / self.grad_accum_every).backward()

        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        if exists(self.vae.discr):
            discr_loss = 0
            for _ in range(self.grad_accum_every):
                img = next(self.dl)
                img = img.to(device)

                loss = self.vae(img, return_discr_loss = True)
                accum_log(logs, {'discr_loss': loss.item() / self.grad_accum_every})

                (loss / self.grad_accum_every).backward()

            self.optim.step()
            self.optim.zero_grad()

            # log

            gen_loss /= self.grad_accum_every
            discr_loss /= self.grad_accum_every

            print(f"{self.steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

        if not (self.steps % self.save_results_every):
            self.vae.eval()
            imgs = next(self.dl)
            imgs = imgs.to(device)

            recons = self.vae(imgs)
            nrows = int(sqrt(self.batch_size))

            imgs_and_recons = torch.stack((imgs, recons), dim = 0)
            imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

            imgs_and_recons = imgs_and_recons.detach().cpu().float()
            grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (-1, 1))

            logs['reconstructions'] = grid

            save_image(grid, str(self.results_folder / f'{self.steps}.png'))

            print(f'{self.steps}: saving to {str(self.results_folder)}')

        if self.steps and not (self.steps % self.save_model_every):
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{self.steps}.pt')
            torch.save(state_dict, model_path)

            print(f'{self.steps}: saving model to {str(self.results_folder)}')

        self.steps += 1
        return logs

    def train(self, log_fn = noop):
        device = next(self.vae.parameters()).device

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        print('training complete')
