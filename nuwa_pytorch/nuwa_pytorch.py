import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from vector_quantize_pytorch import VectorQuantize as VQ
from axial_positional_embedding import AxialPositionalEmbedding

import torchvision

# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# vqgan vae

class Discriminator(nn.Module):
    def __init__(
        self,
        dim,
        num_layers,
        channels = 3
    ):
        super().__init__()
        dims = (channels, *((dim,) * num_layers))
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = MList([])
        for _, (dim_in, dim_out) in zip(range(num_layers), dim_pairs):
            self.layers.append(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1))

        self.to_logits = nn.Sequential(
            Reduce('b d h w -> b d', 'mean'),
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...'),
            nn.Sigmoid()
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

class VQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        num_layers = 3,
        vq_codebook_size = 512,
        vq_decay = 0.8,
        vq_commitment_weight = 1.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.codebook_size = vq_codebook_size

        self.encoders = MList([])
        self.decoders = MList([])

        dims = (channels, *((dim,) * num_layers))
        reversed_dims = tuple(reversed(dims))
        enc_dim_pairs = zip(dims[:-1], dims[1:])
        dec_dim_pairs = zip(reversed_dims[:-1], reversed_dims[1:])

        for _, (enc_dim_in, enc_dim_out), (dec_dim_in, dec_dim_out) in zip(range(num_layers), enc_dim_pairs, dec_dim_pairs):
            self.encoders.append(nn.Conv2d(enc_dim_in, enc_dim_out, 4, stride = 2, padding = 1))
            self.decoders.append(nn.ConvTranspose2d(dec_dim_in, dec_dim_out, 4, stride = 2, padding = 1))

        self.vq = VQ(
            dim = dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True
        )

        self.disc = Discriminator(dim = dim, num_layers = num_layers)

        self.vgg = torchvision.models.vgg16(pretrained = True)
        self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

    def encode(self, img):
        fmap = img

        for enc in self.encoders:
            fmap = enc(fmap)

        return self.vq(fmap)

    @torch.no_grad()
    def get_video_indices(self, video):
        b, f, _, h, w = video.shape
        images = rearrange(video, 'b f ... -> (b f) ...')
        _, indices, _ = self.encode(images)
        return rearrange(indices, '(b f) ... -> b f ...', b = b)

    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False
    ):
        batch, device = img.shape[0], img.device
        fmap = img.clone()

        fmap, indices, commit_loss = self.encode(img)

        for dec in self.decoders:
            fmap = dec(fmap)

        if not return_loss:
            return fmap

        # generator loss

        labels = torch.cat((torch.zeros(batch, device = device), torch.ones(batch, device = device)), dim = 0)

        if return_discr_loss:
            labels = torch.flip(labels, (0,))

        real_or_fake = self.disc(torch.cat((fmap, img), dim = 0))
        gan_loss = F.binary_cross_entropy(real_or_fake, labels)

        if return_discr_loss:
            return gan_loss

        # reconstruction loss

        recon_loss = F.mse_loss(fmap, img)

        # lpips

        img_vgg_feats = self.vgg(img)
        recon_vgg_feats = self.vgg(fmap)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # combine losses

        loss = recon_loss + commit_loss + gan_loss
        return loss

# normalizations

class PreNorm(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fn
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# helper classes

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# attention classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = False
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None
    ):
        h, device = self.heads, x.device

        has_context = exists(context)
        kv_input = context if has_context else x

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(x.dtype).max

        key_mask = mask if not has_context else context_mask

        if exists(key_mask):
            key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~key_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(1)
            sim = sim.masked_fill(mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        cross_attend = False
    ):
        super().__init__()
        self.layers = MList([])
        for _ in range(depth):
            self.layers.append(MList([
                PreNorm(dim = dim, fn = Attention(dim = dim, heads = heads, dim_head = dim_head, causal = causal)),
                PreNorm(dim = dim, fn = Attention(dim = dim, heads = heads, dim_head = dim_head)) if cross_attend else None,
                PreNorm(dim = dim, fn = FeedForward(dim = dim, mult = ff_mult))
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        context_mask = None
    ):
        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask = mask) + x

            if exists(cross_attn):
                x = cross_attn(x, context = context, mask = mask, context_mask = context_mask) + x

            x = ff(x) + x

        return self.norm(x)

# main class

class NUWA(nn.Module):
    def __init__(
        self,
        *,
        vae,
        dim,
        image_size,
        max_video_frames = 5,
        text_num_tokens,
        text_max_seq_len = 256,
        text_enc_depth = 6,
        text_enc_dim_head = 64,
        text_enc_heads = 8,
        dec_depth = 6,
        dec_dim_head = 64,
        dec_heads = 8
    ):
        super().__init__()
        self.vae = vae
        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = nn.Embedding(text_num_tokens, dim)
        self.text_pos_embedding = nn.Embedding(text_max_seq_len, dim)

        self.text_transformer = Transformer(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head
        )

        self.video_bos = nn.Parameter(torch.randn(dim))
        self.image_embedding = nn.Embedding(num_image_tokens, dim)

        fmap_size = image_size // (2 ** vae_num_layers)

        self.video_pos_emb = AxialPositionalEmbedding(
            dim = dim,
            axial_shape = (max_video_frames, fmap_size, fmap_size)
        )

        self.video_transformer = Transformer(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            causal = True,
            cross_attend = True
        )

        self.to_logits = nn.Linear(dim, num_image_tokens)

    def forward(
        self,
        *,
        text,
        image = None,
        video = None,
        text_mask = None,
        return_loss = False
    ):
        batch, seq_len, device = *text.shape, text.device
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        tokens = self.text_embedding(text)
        pos_emb = self.text_pos_embedding(torch.arange(seq_len, device = device))
        tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        text_embeds = self.text_transformer(
            tokens,
            mask = text_mask
        )

        frame_indices = self.vae.get_video_indices(video)
        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        frame_indices_input = frame_indices[:, :-1] if return_loss else frame_indices

        frame_embeddings = self.image_embedding(frame_indices_input)
        frame_embeddings = self.video_pos_emb(frame_embeddings) + frame_embeddings

        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

        frame_embeddings = self.video_transformer(
            frame_embeddings,
            context = text_embeds,
            context_mask = text_mask
        )

        logits = self.to_logits(frame_embeddings)

        if not return_loss:
            return logits

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), frame_indices)
        return loss
