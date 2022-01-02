import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from vector_quantize_pytorch import VectorQuantize as VQ
from axial_positional_embedding import AxialPositionalEmbedding

import torchvision

from unfoldNd import unfoldNd

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
        vq_commitment_weight = 1.,
        l2_recon_loss = False
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

        self.l2_recon_loss = l2_recon_loss

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

        recon_loss_fn = F.mse_loss if self.l2_recon_loss else F.l1_loss
        recon_loss = recon_loss_fn(fmap, img)

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
        causal = False,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
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
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Sparse3DNA(nn.Module):
    def __init__(
        self,
        dim,
        video_shape,
        kernel_size = 3,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel size must be odd'
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.kernel_size = kernel_size
        self.video_shape = video_shape

        max_num_tokens = torch.empty(video_shape).numel()
        self.max_num_tokens = max_num_tokens

        # precalculate causal mask

        indices = torch.arange(max_num_tokens)
        shaped_indices = rearrange(indices, '(f h w) -> 1 1 f h w', f = video_shape[0], h = video_shape[1], w = video_shape[2])
        padded_indices = F.pad(shaped_indices, (kernel_size // 2,) * 6, value = max_num_tokens) # padding has value of max tokens so to be masked out
        unfolded_indices = unfoldNd(padded_indices, kernel_size = kernel_size)
        unfolded_indices = rearrange(unfolded_indices, '1 k n -> n k')

        causal_mask = rearrange(indices, 'n -> n 1') < unfolded_indices
        causal_mask = F.pad(causal_mask, (1, 0), value = False) # bos tokens never get masked out
        self.register_buffer('causal_mask', causal_mask)

    def forward(self, x, mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device

        # more variables

        kernel_size = self.kernel_size
        num_frames, fmap_size, _ = self.video_shape

        # pad for last token in video

        x = F.pad(x, (0, 0, 0, 1), value = 0.)

        # derive queries / keys / values

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # scale queries

        q = q * self.scale

        # take care of bos

        q = q[:, 1:]
        bos_value = v[:, :1]

        # prepare precomputed causal mask

        causal_mask = self.causal_mask[:n]
        causal_mask = repeat(causal_mask, 'i j -> b i j', b = b * h)

        # compute keys and values

        (k_bos, k), (v_bos, v) = map(lambda t: (t[:, :1], t[:, 1:]), (k, v))
        k, v = map(lambda t: rearrange(t, 'b (f h w) d -> b d f h w', f  = num_frames, h = fmap_size), (k, v))
        k, v = map(lambda t: unfoldNd(t, kernel_size = kernel_size, padding = kernel_size // 2), (k, v))
        k, v = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = kernel_size ** 3), (k, v))

        # append bos keys and values

        k_bos, v_bos = map(lambda t: repeat(t, 'b 1 d -> b n 1 d', n = k.shape[1]), (k_bos, v_bos))
        k = torch.cat((k_bos, k), dim = 2)
        v = torch.cat((v_bos, v), dim = 2)

        # calculate sim

        sim = einsum('b i d, b i j d -> b i j', q, k)

        # causal mask

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b i j, b i j d -> b i d', attn, v)

        # append bos value

        out = torch.cat((bos_value, out), dim = 1)  # bos will always adopt its own value, since it pays attention only to itself

        # merge heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out[:, :n])

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
        cross_attend = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        sparse_3dna_attn = False,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_video_shape = None
    ):
        super().__init__()
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'

        self.layers = MList([])
        for _ in range(depth):
            if sparse_3dna_attn:
                self_attn = Sparse3DNA(dim = dim, heads = heads, dim_head = dim_head, kernel_size = sparse_3dna_kernel_size, video_shape = sparse_3dna_video_shape)
            else:
                self_attn = Attention(dim = dim, heads = heads, dim_head = dim_head, causal = causal, dropout = attn_dropout)

            self.layers.append(MList([
                PreNorm(dim = dim, fn = self_attn),
                PreNorm(dim = dim, fn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)) if cross_attend else None,
                PreNorm(dim = dim, fn = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
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
        dec_heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        sparse_3dna_kernel_size = 3
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
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
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
            cross_attend = True,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            sparse_3dna_attn = True,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_video_shape = (max_video_frames, fmap_size, fmap_size)
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
