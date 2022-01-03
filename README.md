<img src="./nuwa.png" width="400px"></img>

## NÜWA - Pytorch (wip)

<a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a></br>

Implementation of <a href="https://arxiv.org/abs/2111.12417">NÜWA</a>, state of the art attention network for text to video synthesis, in Pytorch. This repository will be populated in the case that Microsoft does not open source the code by end of December. It may also contain an extension into video and audio, using a dual decoder approach.

<a href="https://www.youtube.com/watch?v=InhMx1h0N40">Yannic Kilcher</a>

<a href="https://www.youtube.com/watch?v=C9CTnZJ9ZE0">DeepReader</a>

## Install

```bash
$ pip install nuwa-pytorch
```

## Usage

First train the VAE

```python
import torch
from nuwa_pytorch import VQGanVAE

vae = VQGanVAE(
    dim = 512,
    image_size = 256,
    num_layers = 4
)

imgs = torch.randn(10, 3, 256, 256)

# alternate learning for autoencoder ...

loss = vae(imgs, return_loss = True)
loss.backward()

# and the discriminator ...

discr_loss = vae(imgs, return_discr_loss = True)
discr_loss.backward()

# do above for many steps
```

Then, with your learned VAE

```python
import torch
from nuwa_pytorch import NUWA, VQGanVAE

# autoencoder

vae = VQGanVAE(
    dim = 512,
    num_layers = 4,
    image_size = 256
)

# NUWA transformer

nuwa = NUWA(
    vae = vae,
    dim = 512,
    max_video_frames = 5,
    text_num_tokens = 20000,
    image_size = 256
).cuda()

# data

text = torch.randint(0, 20000, (1, 256)).cuda()
mask = torch.ones(1, 256).bool().cuda()
video = torch.randn(1, 5, 3, 256, 256).cuda()

loss = nuwa(
    text = text,
    video = video,
    text_mask = mask,
    return_loss = True
)

loss.backward()

# do above with as much data as possible

# then you can generate a video from text

video = nuwa.generate(text = text, text_mask = mask) # (1, 5, 3, 256, 256)

```

## Todo

- [x] complete 3dna causal attention in decoder
- [x] write up easy generation functions
- [ ] flesh out VAE resnet blocks, offer some choices
- [ ] make sure GAN portion of VQGan is correct, reread paper
- [ ] offer new vqvae improvements (orthogonal reg and smaller codebook dimensions)
- [ ] offer vqvae training script
- [ ] take care of audio transformer and cross modality attention
- [ ] segmentation mask encoder, make sure embeddings can undergo 3dna attention with decoder during cross attention
- [ ] investigate custom attention layouts in microsoft deepspeed sparse attention (using triton)

## Citations

```bibtex
@misc{wu2021nuwa,
    title   = {N\"UWA: Visual Synthesis Pre-training for Neural visUal World creAtion}, 
    author  = {Chenfei Wu and Jian Liang and Lei Ji and Fan Yang and Yuejian Fang and Daxin Jiang and Nan Duan},
    year    = {2021},
    eprint  = {2111.12417},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
