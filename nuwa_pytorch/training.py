from math import sqrt
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW, Adam

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image

from einops import rearrange
from nuwa_pytorch.nuwa_pytorch import VQGanVAE

# helpers

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

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

    params = set(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': list(wd_params)},
        {'params': list(no_wd_params), 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = wd)

# classes

class VQGanVAETrainer(nn.Module):
    def __init__(
        self,
        vae,
        *,
        folder,
        num_train_steps,
        lr,
        batch_size,
        grad_accum_every,
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

        self.optim = Adam(vae.parameters(), lr = lr)

        self.ds = ImageFolder(
            folder,
            T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor()
            ])
        )

        self.dl = cycle(DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        ))

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents = True, exist_ok = True)

    def train(self):
        device = next(self.vae.parameters()).device

        while self.steps < self.num_train_steps:
            self.vae.train()

            # update vae (generator)

            gen_loss = 0
            for _ in range(self.grad_accum_every):
                img, _ = next(self.dl)
                img = img.to(device)

                loss = self.vae(img, return_loss = True)
                gen_loss += loss.item()

                (loss / self.grad_accum_every).backward()

            self.optim.step()
            self.optim.zero_grad()

            # update discriminator

            discr_loss = 0
            for _ in range(self.grad_accum_every):
                img, _ = next(self.dl)
                img = img.to(device)

                loss = self.vae(img, return_discr_loss = True)
                discr_loss += loss.item()

                (loss / self.grad_accum_every).backward()

            self.optim.step()
            self.optim.zero_grad()

            # log

            gen_loss /= self.grad_accum_every
            discr_loss /= self.grad_accum_every

            print(f'{self.steps}: vae loss: {gen_loss} - discr loss: {discr_loss}')

            if not (self.steps % self.save_results_every):
                self.vae.eval()
                imgs, _ = next(self.dl)
                imgs = imgs.to(device)

                recons = self.vae(imgs)
                nrows = int(sqrt(self.batch_size))

                imgs_and_recons = torch.stack((imgs, recons), dim = 0)
                imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

                imgs_and_recons = imgs_and_recons.detach().cpu().float()
                grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (-1, 1))

                save_image(grid, str(self.results_folder / f'{self.steps}.png'))
                print(f'saving to {str(self.results_folder)}')

            if self.steps and not (self.steps % self.save_model_every):
                state_dict = self.vae.state_dict()
                model_path = str(self.results_folder / f'vae.{self.steps}.pt')
                torch.save(state_dict, model_path)

                print(f'saving model to {str(self.results_folder)}')

            self.steps += 1

        print('training complete')
