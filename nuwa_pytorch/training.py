import torch
from torch import nn
from torch.optim import AdamW, Adam

from nuwa_pytorch.nuwa_pytorch import VQGanVAE

# helper functions

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
        *,
        vae,
        lr = 3e-4
    ):
        super().__init__()
        assert isinstance(vae, VQGanVAE), 'vae must be instance of VQGanVAE'

        self.vae = vae
        self.optim = Adam(vae.parameters(), lr = lr)
        self.register_buffer('state', torch.ones((1,), dtype = torch.bool))

    def forward(self, img):
        return_loss_key = 'return_loss' if self.state else 'return_discr_loss'
        vae_kwargs = {return_loss_key: True}

        loss = self.vae(img, **vae_kwargs)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()

        self.state = self.state.data.copy_(~self.state)
        return loss, bool(self.state)
