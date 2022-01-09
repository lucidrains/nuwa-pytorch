<img src="./nuwa.png" width="400px"></img>

## NÜWA - Pytorch

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
    text_num_tokens = 20000,           # number of text tokens
    text_enc_depth = 12,               # text encoder depth
    text_enc_heads = 8,                # number of attention heads for encoder
    text_max_seq_len = 256,            # max sequence length of text conditioning tokens (keep at 256 as in paper, or shorter, if your text is not that long)
    max_video_frames = 10,             # number of video frames
    image_size = 256,                  # size of each frame of video
    dec_depth = 64,                    # video decoder depth
    dec_heads = 8,                     # number of attention heads in decoder
    dec_reversible = True,             # reversible networks - from reformer, decoupling memory usage from depth
    enc_reversible = True,             # reversible encoders, if you need it
    attn_dropout = 0.05,               # dropout for attention
    ff_dropout = 0.05,                 # dropout for feedforward
    sparse_3dna_kernel_size = 3,       # kernel size of the sparse 3dna attention
    sparse_3dna_dilation = (1, 2, 4),  # cycle dilation of 3d conv attention in decoder, for more range
    shift_video_tokens = True          # cheap relative positions for sparse 3dna transformer, by shifting along spatial dimensions by one
).cuda()

# data

text = torch.randint(0, 20000, (1, 256)).cuda()
video = torch.randn(1, 10, 3, 256, 256).cuda() # (batch, frames, channels, height, width)

loss = nuwa(
    text = text,
    video = video,
    return_loss = True  # set this to True, only for training, to return cross entropy loss
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
- [x] add feedforward chunking
- [x] add shift token in decoder for cheap powerful RPE
- [x] add reversible networks, to save on memory on depth
- [ ] add cosine sim attention from swinv2 as an option
- [ ] offer vqvae training script
- [ ] take care of audio transformer and cross modality attention
- [ ] segmentation mask encoder, make sure embeddings can undergo 3dna attention with decoder during cross attention
- [ ] add audio transformer, and build audio / video nearby cross attention
- [ ] add some autotrainer that takes care of the alternating updates of discriminator and VQVAE generator
- [ ] allow for variable lengthed videos in sparse 3dna non-causal attention
- [ ] support kernel sizes different along each dimension for sparse 3dna
- [ ] Triton kernel for 3dna attention
- [ ] offer a colab with moving mnist example, conditioned on present digits
- [ ] rotary embeddings for encoder

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
@misc{kitaev2020reformer,
    title   = {Reformer: The Efficient Transformer},
    author  = {Nikita Kitaev and Łukasz Kaiser and Anselm Levskaya},
    year    = {2020},
    eprint  = {2001.04451},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
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
