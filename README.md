<img src="./nuwa.png" width="400px"></img>

## NÜWA - Pytorch (wip)

<a href="https://discord.gg/xBPBXfcFHd"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a></br>

Implementation of <a href="https://arxiv.org/abs/2111.12417">NÜWA</a>, state of the art attention network for text to video synthesis, in Pytorch. It may also contain an extension into video and audio, using a dual decoder approach.

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
video = torch.randn(1, 5, 3, 256, 256).cuda() # (batch, frames, channels, height, width)

loss = nuwa(
    text = text,
    video = video,
    return_loss = True
)

loss.backward()

# do above with as much data as possible

# then you can generate a video from text

video = nuwa.generate(text = text) # (1, 5, 3, 256, 256)

```

## Todo

- [x] complete 3dna causal attention in decoder
- [x] write up easy generation functions
- [x] make sure GAN portion of VQGan is correct, reread paper
- [x] make sure adaptive weight in vqgan is correctly built
- [x] offer new vqvae improvements (orthogonal reg and smaller codebook dimensions)
- [x] batch video tokens -> vae during video generation, to prevent oom
- [x] query chunking in 3dna attention, to put a cap on peak memory
- [x] flesh out VAE resnet blocks, offer some choices
- [x] add all stability tricks from cogview paper by default
- [x] make VQGan able to accept custom VGG for LPAPs loss (audio)
- [ ] add cosine sim attention from swinv2 as an option
- [ ] offer vqvae training script
- [ ] take care of audio transformer and cross modality attention
- [ ] segmentation mask encoder, make sure embeddings can undergo 3dna attention with decoder during cross attention
- [ ] add audio transformer, and build audio / video nearby cross attention
- [ ] add some autotrainer that takes care of the alternating updates of discriminator and VQVAE generator
- [ ] add reversible networks and feedforward chunking, a la Reformer, to save on memory
- [ ] allow for variable lengthed videos in sparse 3dna non-causal attention
- [ ] add shift token in decoder for cheap powerful RPE

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

```bibtex
@misc{ding2021cogview,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    year    = {2021},
    eprint  = {2105.13290},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@inproceedings{ho2021classifierfree,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho and Tim Salimans},
    booktitle = {NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications},
    year    = {2021},
    url     = {https://openreview.net/forum?id=qw8AKxfYbI}
}
```

```bibtex
@misc{crowson2022,
    author  = {Katherine Crowson},
    url     = {https://twitter.com/RiversHaveWings/status/1478093658716966912}
}
```
