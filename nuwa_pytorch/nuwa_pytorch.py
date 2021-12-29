import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from vector_quantize_pytorch import VectorQuantize as VQ

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# vqgan vae

class VQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim
    ):
        super().__init__()

    def forward(
        self,
        img
    ):
        return img

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
        return self.fn(self.norm(x), **kwargs)

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
        dim_head = 64
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
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
        heads = 8,
        dim_head = 64,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim = dim, fn = Attention(dim = dim, heads = heads, dim_head = dim_head)),
                PreNorm(dim = dim, fn = FeedForward(dim = dim, mult = ff_mult))
            ]))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return self.norm(x)

# main class

class NUWA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        text_num_tokens,
        text_max_seq_len = 256,
        text_enc_depth = 6,
        text_enc_dim_head = 64,
        text_enc_heads = 8
    ):
        super().__init__()
        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = nn.Embedding(text_num_tokens, dim)
        self.text_pos_embedding = nn.Embedding(text_max_seq_len, dim)

        self.text_transformer = Transformer(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head
        )

    def forward(
        self,
        *,
        text,
        image = None,
        video = None,
        text_mask = None,
        return_loss = False
    ):
        seq_len, device = text.shape[1], text.device
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        tokens = self.text_embedding(text)
        pos_emb = self.text_pos_embedding(torch.arange(seq_len, device = device))
        tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        tokens = self.text_transformer(tokens)
        return tokens
