from functools import partial
import torch
from torch import nn, einsum
from torch.autograd import grad
from torch.optim import Adam
import torch.nn.functional as F
import torchvision

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from vector_quantize_pytorch import VectorQuantize as VQ

from nuwa_pytorch.reversible import ReversibleSequence

from unfoldNd import unfoldNd

# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, size = 1):
    return val if isinstance(val, tuple) else (val,) * size

def calc_same_padding(kernel_size, dilation = 1):
    return dilation * (kernel_size - 1) // 2

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helper functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def sigmoid(t):
    return torch.where(t >= 0, 1 / (1 + torch.exp(-t)), t.exp() / (1 + t.exp()))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def safe_div(numer, denom, eps = 1e-6):
    return numer / (denom + eps)

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    return (t * alpha).softmax(dim = dim)

def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - sigmoid(fake)) - log(sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

def batch_process(t, fn, chunks = 10, dim = 0):
    chunks = [fn(t_chunk) for t_chunk in t.chunk(chunks, dim = dim)]
    return torch.cat(chunks, dim = dim)

# gradient control

def frac_gradient(t, frac):
    return t * frac + t.detach() * (1 - frac)

# vqgan vae

class LayerNormChan(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

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
            Rearrange('... 1 -> ...')
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

class ConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        act = nn.ReLU(),
        ds_kernel_size = 7,
        mult = 4,
    ):
        """
        https://arxiv.org/abs/2201.03545
        """
        super().__init__()
        inner_dim = dim * mult

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, ds_kernel_size, padding = ds_kernel_size // 2, groups = dim),
            LayerNormChan(dim),
            nn.Conv2d(dim, inner_dim, 1),
            act,
            nn.Conv2d(inner_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class VQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        num_layers = 4,
        vq_codebook_size = 512,
        vq_decay = 0.8,
        vq_commitment_weight = 1.,
        l2_recon_loss = False,
        use_hinge_loss = True,
        num_conv_blocks = 1,
        vgg = None,
        **kwargs
    ):
        super().__init__()
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)

        self.num_layers = num_layers
        self.codebook_size = vq_codebook_size

        self.encoders = MList([])
        self.decoders = MList([])

        dims = (dim,) * (num_layers + 1)
        reversed_dims = tuple(reversed(dims))
        enc_dim_pairs = zip(dims[:-1], dims[1:])
        dec_dim_pairs = zip(reversed_dims[:-1], reversed_dims[1:])

        for _, (enc_dim_in, enc_dim_out), (dec_dim_in, dec_dim_out) in zip(range(num_layers), enc_dim_pairs, dec_dim_pairs):
            self.encoders.append(nn.Sequential(nn.Conv2d(enc_dim_in, enc_dim_out, 4, stride = 2, padding = 1), nn.ReLU()))
            self.decoders.append(nn.Sequential(nn.ConvTranspose2d(dec_dim_in, dec_dim_out, 4, stride = 2, padding = 1), nn.ReLU()))

        for _ in range(num_conv_blocks):
            self.encoders.append(ConvNextBlock(dims[-1]))
            self.decoders.append(ConvNextBlock(dims[-1]))

        self.encoders.insert(0, nn.Conv2d(channels, dim, 3, padding = 1))
        self.decoders.append(nn.Conv2d(dim, channels, 1))

        self.vq = VQ(
            dim = dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True,
            **vq_kwargs
        )

        # reconstruction loss

        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # preceptual loss

        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # gan related losses

        self.discr = Discriminator(dim = dim, num_layers = num_layers)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, fmap):
        for enc in self.encoders:
            fmap = enc(fmap)

        return self.vq(fmap)

    def decode(self, fmap):
        for dec in self.decoders:
            fmap = dec(fmap)

        return fmap

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

        fmap = self.decode(fmap)

        if not return_loss and not return_discr_loss:
            return fmap

        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # whether to return discriminator loss

        if return_discr_loss:
            fmap.detach_()
            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))
            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)
            return discr_loss

        # perceptual loss

        img_vgg_feats = self.vgg(img)
        recon_vgg_feats = self.vgg(fmap)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(fmap)

        # calculate adaptive weight

        last_dec_layer = self.decoders[-1].weight

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # reconstruction loss

        recon_loss = self.recon_loss_fn(fmap, img)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss
        return loss

# normalizations

class StableLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x / x.amax(dim = -1, keepdim = True).detach()
        return self.norm(x)

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

class SandwichNorm(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fn
    ):
        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.postnorm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.prenorm(x)
        x = self.fn(x, **kwargs)
        x = self.postnorm(x)
        return x

# helper classes

class ShiftVideoTokens(nn.Module):
    def __init__(
        self,
        fn,
        image_size,
        shift_space = True,
        shift_time = False
    ):
        super().__init__()
        self.fn = fn
        self.image_size = image_size

        self.shift_time = shift_time
        self.shift_space = shift_space

    def forward(self, x, **kwargs):

        if not self.shift_time and not self.shift_space:
            return self.fn(x, **kwargs)

        image_size = self.image_size
        img_seq_len = image_size ** 2

        x_bos, x_video = x[:, :1], x[:, 1:]
        n = x_video.shape[1]

        # pad to nearest frame

        padding = img_seq_len - (n % img_seq_len)
        x_video = F.pad(x_video, (0, 0, 0, padding), value = 0.)

        # reshape to video

        x_video = rearrange(x_video, 'b (f h w) d -> b f h w d', h = image_size, w = image_size)

        x_image_h = x_image_w = x_frame = None

        # chunk depending on whether shifting time, space, or both

        if self.shift_space and self.shift_time:
            x_frame, x_image_h, x_image_w, *x_rest = x_video.chunk(5, dim = -1)
        elif self.shift_space:
            x_image_h, x_image_w, *x_rest = x_video.chunk(4, dim = -1)
        elif self.shift_time:
            x_frame, *x_rest = x_video.chunk(3, dim = -1)

        # shifts

        if self.shift_space:
            x_image_h = F.pad(x_image_h, (0, 0, 0, 0, 1, -1))
            x_image_w = F.pad(x_image_w, (0, 0, 1, -1))

        if self.shift_time:
            x_frame = F.pad(x_frame, (0, 0, 0, 0, 0, 0, 1, -1))

        # concat

        x_shifted = [x_frame, x_image_h, x_image_w, *x_rest]
        x_shifted = list(filter(exists, x_shifted))

        x_video = torch.cat(x_shifted, dim = -1)

        # merge text and image sequence back together

        x_video = rearrange(x_video, 'b f h w d -> b (f h w) d')
        x_video = x_video[:, :n]

        x = torch.cat((x_bos, x_video), dim = 1)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.,
        chunk_size = None,  # chunk size to process feedforward, along sequence length, from Reformer paper. None means do not chunk
    ):
        super().__init__()
        inner_dim = (dim * mult * 2) // 3
        self.chunk_size = chunk_size

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        if not exists(self.chunk_size):
            return self.net(x)

        x_chunks = x.split(self.chunk_size, dim = -2)
        out_chunks = [self.net(c) for c in x_chunks]
        return torch.cat(out_chunks, dim = -2)

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

        self.null_k = nn.Parameter(torch.randn(1, heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(1, heads, 1, dim_head))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
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

        # add null key / values, needed for condition dropout

        k = torch.cat((self.null_k, k), dim = -2)
        v = torch.cat((self.null_v, v), dim = -2)

        # scale

        q = q * self.scale

        # similarity

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        mask_value = -torch.finfo(x.dtype).max

        key_mask = mask if not has_context else context_mask

        if exists(key_mask):
            key_mask = F.pad(key_mask, (1, 0), value = True) # always pay attention to null key / value
            key_mask = rearrange(key_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~key_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(j - i + 1)
            sim = sim.masked_fill(mask, mask_value)

        # attention

        attn = stable_softmax(sim, dim = -1)
        attn = self.talking_heads(attn)
        attn = self.dropout(attn)

        # aggregate, merge, and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Sparse3DNA(nn.Module):
    def __init__(
        self,
        dim,
        video_shape,
        kernel_size = 3,
        dilation = 1,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False,
        query_num_frames_chunk = None
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dilation = cast_tuple(dilation, size = 3)

        self.kernel_size = cast_tuple(kernel_size, size = 3)
        assert all(map(lambda n: n % 2 == 1, self.kernel_size)), 'kernel size must be odd'

        self.kernel_numel = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]

        # calculate padding

        self.padding_frame = calc_same_padding(self.kernel_size[0], self.dilation[0])
        self.padding_height = calc_same_padding(self.kernel_size[1], self.dilation[1])
        self.padding_width = calc_same_padding(self.kernel_size[2], self.dilation[2])

        self.video_padding = (self.padding_width, self.padding_width, self.padding_height, self.padding_height, self.padding_frame, self.padding_frame)

        # save video shape and calculate max number of tokens

        self.video_shape = video_shape
        max_frames, fmap_size, _ = video_shape
        max_num_tokens = torch.empty(video_shape).numel()
        self.max_num_tokens = max_num_tokens

        # how many query tokens to process at once to limit peak memory usage, by multiple of frame tokens (fmap_size ** 2)

        self.query_num_frames_chunk = default(query_num_frames_chunk, max_frames)

        # precalculate causal mask

        indices = torch.arange(max_num_tokens)
        shaped_indices = rearrange(indices, '(f h w) -> 1 1 f h w', f = max_frames, h = fmap_size, w = fmap_size)
        padded_indices = F.pad(shaped_indices, self.video_padding, value = max_num_tokens) # padding has value of max tokens so to be masked out
        unfolded_indices = unfoldNd(padded_indices, kernel_size = self.kernel_size, dilation = self.dilation)
        unfolded_indices = rearrange(unfolded_indices, '1 k n -> n k')

        # if causal, compare query and key indices and make sure past cannot see future
        # if not causal, just mask out the padding

        if causal:
            mask = rearrange(indices, 'n -> n 1') < unfolded_indices
        else:
            mask = unfolded_indices == max_num_tokens

        mask = F.pad(mask, (1, 0), value = False) # bos tokens never get masked out
        self.register_buffer('mask', mask)

    def forward(self, x):
        b, n, _, h, device = *x.shape, self.heads, x.device

        # more variables

        dilation = self.dilation
        kernel_size = self.kernel_size
        video_padding = self.video_padding
        fmap_size = self.video_shape[1]

        bos_only = n == 1
        tokens_per_frame = fmap_size ** 2

        padding = 0 if bos_only else (tokens_per_frame - (n - 1) % tokens_per_frame)
        num_frames = (n + padding) // tokens_per_frame

        # pad for last token in video

        padded_x = F.pad(x, (0, 0, 0, padding), value = 0.) if padding > 0 else x

        # derive queries / keys / values

        q, k, v = (self.to_q(x), *self.to_kv(padded_x).chunk(2, dim = -1))

        # early return if <bos>

        if bos_only:
            return self.to_out(v)

        # split out heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        # scale queries

        q = q * self.scale

        # take care of bos

        q = q[:, 1:]
        bos_value = v[:, :1]

        # compute keys and values

        (k_bos, k), (v_bos, v) = map(lambda t: (t[:, :1], t[:, 1:]), (k, v))

        # reshape keys and values to video and add appropriate padding along all dimensions (frames, height, width)

        k, v = map(lambda t: rearrange(t, 'b (f h w) d -> b d f h w', f  = num_frames, h = fmap_size), (k, v))
        k, v = map(lambda t: F.pad(t, video_padding), (k, v))

        # put the attention processing code in a function
        # to allow for processing queries in chunks of frames

        out = []

        def attend(q, k, v, mask, k_bos, v_bos, kernel_size):
            chunk_length = q.shape[1]

            k, v = map(lambda t: unfoldNd(t, kernel_size = kernel_size, dilation = dilation), (k, v))
            k, v = map(lambda t: rearrange(t, 'b (d j) i -> b i j d', j = self.kernel_numel), (k, v))
            k, v = map(lambda t: t[:, :chunk_length], (k, v))

            # append bos keys and values

            k_bos, v_bos = map(lambda t: repeat(t, 'b 1 d -> b n 1 d', n = k.shape[1]), (k_bos, v_bos))
            k = torch.cat((k_bos, k), dim = 2)
            v = torch.cat((v_bos, v), dim = 2)

            # calculate sim

            sim = einsum('b i d, b i j d -> b i j', q, k)

            # causal mask

            if exists(mask):
                mask_value = -torch.finfo(sim.dtype).max
                mask = rearrange(mask, 'i j -> 1 i j')
                sim = sim.masked_fill(mask, mask_value)

            # attention

            attn = sim.softmax(dim = -1)

            attn = rearrange(attn, '(b h) ... -> b h ...', h = h)
            attn = self.talking_heads(attn)
            attn = rearrange(attn, 'b h ... -> (b h) ...')

            attn = self.dropout(attn)

            # aggregate values

            return einsum('b i j, b i j d -> b i d', attn, v)

        # process queries in chunks

        frames_per_chunk = min(self.query_num_frames_chunk, num_frames)
        chunk_size = frames_per_chunk * tokens_per_frame

        q_chunks = q.split(chunk_size, dim = 1)

        mask = self.mask[:(n - 1)]
        mask_chunks = mask.split(chunk_size, dim = 0)

        for ind, (q_chunk, mask_chunk) in enumerate(zip(q_chunks, mask_chunks)):
            q_chunk = q_chunks[ind]
            mask_chunk = mask_chunks[ind]

            # slice the keys and values to the appropriate frames, accounting for padding along frames dimension

            kv_start_pos = ind * frames_per_chunk
            kv_end_pos = kv_start_pos + (ind + frames_per_chunk + self.padding_frame * 2)
            kv_frame_range = slice(kv_start_pos, kv_end_pos)

            k_slice, v_slice = map(lambda t: t[:, :, kv_frame_range], (k, v))

            # calculate output chunk

            out_chunk = attend(
                q = q_chunk,
                k = k_slice,
                v = v_slice,
                mask = mask_chunk,
                k_bos = k_bos,
                v_bos = v_bos,
                kernel_size = kernel_size,
            )

            out.append(out_chunk)

        # combine all chunks

        out = torch.cat(out, dim = 1)

        # append bos value

        out = torch.cat((bos_value, out), dim = 1)  # bos will always adopt its own value, since it pays attention only to itself

        # merge heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
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
        cross_attend = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        sparse_3dna_attn = False,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_video_shape = None,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        shift_video_tokens = False
    ):
        super().__init__()
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'

        self.layers = MList([])

        for ind in range(depth):
            if sparse_3dna_attn:
                dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]

                self_attn = Sparse3DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    kernel_size = sparse_3dna_kernel_size,
                    dilation = dilation,
                    video_shape = sparse_3dna_video_shape,
                    query_num_frames_chunk = sparse_3dna_query_num_frames_chunk
                )
            else:
                self_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    dropout = attn_dropout
                )

            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            if sparse_3dna_attn and shift_video_tokens:
                fmap_size = sparse_3dna_video_shape[-1]
                self_attn = ShiftVideoTokens(self_attn, image_size = fmap_size)
                ff        = ShiftVideoTokens(ff, image_size = fmap_size)

            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = self_attn),
                SandwichNorm(dim = dim, fn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)) if cross_attend else None,
                SandwichNorm(dim = dim, fn = ff)
            ]))

        self.norm = StableLayerNorm(dim)

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

class ReversibleTransformer(nn.Module):
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
        ff_chunk_size = None,
        sparse_3dna_attn = False,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_video_shape = None,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        shift_video_tokens = False
    ):
        super().__init__()
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'
        self.layers = MList([])

        for ind in range(depth):
            if sparse_3dna_attn:
                dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]
                image_size = sparse_3dna_video_shape[-1]

                self_attn = Sparse3DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    kernel_size = sparse_3dna_kernel_size,
                    dilation = dilation,
                    video_shape = sparse_3dna_video_shape,
                    query_num_frames_chunk = sparse_3dna_query_num_frames_chunk
                )
            else:
                image_size = None

                self_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    causal = causal,
                    dropout = attn_dropout
                )

            wrapper_fn = partial(ShiftVideoTokens, image_size = image_size, shift_space = sparse_3dna_attn and shift_video_tokens)

            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = wrapper_fn(self_attn)),
                SandwichNorm(dim = dim, fn = wrapper_fn(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)))
            ]))

            if not cross_attend:
                continue

            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                SandwichNorm(dim = dim, fn = wrapper_fn(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)))
            ]))

        attn_context_layer = ((True, False),) if cross_attend else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        attn_route_map = {'mask': route_attn}

        self.net = ReversibleSequence(self.layers, args_route = {**context_route_map, **attn_route_map})
        self.norm = StableLayerNorm(dim)

    def forward(
        self,
        x,
        **kwargs
    ):
        x = self.net(x, **kwargs)
        return self.norm(x)

# embeddings

class Embedding(nn.Module):
    def __init__(self, *shape, frac_gradient = 1.):
        super().__init__()
        self.frac_gradient = frac_gradient
        self.embed = nn.Embedding(*shape)

    def forward(self, x):
        x = self.embed(x)

        if self.training and self.frac_gradient < 1:
            x = frac_gradient(x, self.frac_gradient)

        return x

# positional embedding

class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        *,
        shape
    ):
        super().__init__()
        self.dim = dim
        frames, height, width = shape
        self.pos_frames = nn.Parameter(torch.randn(frames, dim))
        self.pos_height = nn.Parameter(torch.randn(height, dim))
        self.pos_width = nn.Parameter(torch.randn(width, dim))

    def forward(self):
        pos_frames = rearrange(self.pos_frames, 'f d -> f 1 1 d')
        pos_height = rearrange(self.pos_height, 'h d -> 1 h 1 d')
        pos_width = rearrange(self.pos_width, 'w d -> 1 1 w d')
        positions = pos_frames + pos_height + pos_width
        return rearrange(positions, 'f h w d -> 1 (f h w) d')

# sampling helpers

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

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
        enc_reversible = False,
        dec_depth = 6,
        dec_dim_head = 64,
        dec_heads = 8,
        dec_reversible = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        embed_gradient_frac = 0.2,
        shift_video_tokens = True,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilation = 1,
    ):
        super().__init__()
        self.vae = vae
        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = Embedding(text_num_tokens, dim, frac_gradient = embed_gradient_frac)
        self.text_pos_embedding = Embedding(text_max_seq_len, dim)

        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        self.text_transformer = enc_transformer_klass(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        self.video_bos = nn.Parameter(torch.randn(dim))
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        fmap_size = image_size // (2 ** vae_num_layers)

        self.video_fmap_size = fmap_size
        self.max_video_frames = max_video_frames
        video_shape = (max_video_frames, fmap_size, fmap_size)

        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # cycle dilation for sparse 3d-nearby attention

        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        dec_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        self.video_transformer = dec_transformer_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            causal = True,
            cross_attend = True,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_attn = True,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk
        )

        self.to_logits = nn.Linear(dim, num_image_tokens)

    def embed_text(self, text, mask = None):
        batch, seq_len, device = *text.shape, text.device
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        tokens = self.text_embedding(text)
        pos_emb = self.text_pos_embedding(torch.arange(seq_len, device = device))
        tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        return self.text_transformer(
            tokens,
            mask = mask
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        text,
        filter_thres = 0.9,
        temperature = 1.,
        decode_max_batchsize = 10,
        cond_scale = 2.,
        num_frames = None
    ):
        batch, seq_len, device = *text.shape, text.device

        text_mask = text != 0
        text_embeds = self.embed_text(text, mask = text_mask)

        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)

        video_indices = torch.empty((batch, 0), device = device, dtype = torch.long)

        num_tokens_per_frame = self.video_fmap_size ** 2

        num_frames = default(num_frames, self.max_video_frames)
        total_video_tokens =  num_tokens_per_frame * num_frames
        max_video_tokens = num_tokens_per_frame * self.max_video_frames

        pos_emb = self.video_pos_emb()

        for ind in range(total_video_tokens):
            video_indices_input = video_indices

            num_video_tokens = video_indices.shape[1]
            if num_video_tokens > max_video_tokens:
                curr_frame_tokens = num_video_tokens % num_tokens_per_frame
                lookback_tokens = (self.max_video_frames - (0 if curr_frame_tokens == 0 else 1)) * num_tokens_per_frame + curr_frame_tokens
                video_indices_input = video_indices[:, -lookback_tokens:]

            frame_embeddings = self.image_embedding(video_indices_input)
            frame_embeddings = pos_emb[:, :frame_embeddings.shape[1]] + frame_embeddings
            frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

            frame_embeddings = self.video_transformer(
                frame_embeddings,
                context = text_embeds,
                context_mask = text_mask
            )

            logits = self.to_logits(frame_embeddings)

            if cond_scale != 1:
                # discovery by Katherine Crowson
                # https://twitter.com/RiversHaveWings/status/1478093658716966912
                uncond_frame_embeddings = self.video_transformer(
                    frame_embeddings,
                    context = text_embeds,
                    context_mask = torch.zeros_like(text_mask).bool()
                )

                uncond_logits = self.to_logits(uncond_frame_embeddings)
                logits = uncond_logits + (logits - uncond_logits) * cond_scale

            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sample = rearrange(sample, 'b -> b 1')
            video_indices = torch.cat((video_indices, sample), dim = 1)

        codes = self.vae.codebook[video_indices]
        codes = rearrange(codes, 'b (f h w) d -> (b f) d h w', h = self.video_fmap_size, w = self.video_fmap_size)

        image_reconstructions = batch_process(codes, self.vae.decode, chunks = decode_max_batchsize)
        video = rearrange(image_reconstructions, '(b f) d h w -> b f d h w', b = batch)
        return video

    def forward(
        self,
        *,
        text,
        image = None,
        video = None,
        return_loss = False,
        cond_dropout_prob = 0.2
    ):
        batch, seq_len, frames, device = *text.shape, video.shape[1], text.device

        text_mask = text != 0
        text_embeds = self.embed_text(text, mask = text_mask)

        assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'

        frame_indices = self.vae.get_video_indices(video)
        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        frame_indices_input = frame_indices[:, :-1] if return_loss else frame_indices

        frame_embeddings = self.image_embedding(frame_indices_input)
        frame_embeddings = self.video_pos_emb()[:, :-1] + frame_embeddings

        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        frame_embeddings = torch.cat((bos, frame_embeddings), dim = 1)

        if self.training and cond_dropout_prob > 0:
            # dropout condition randomly
            # presented in https://openreview.net/forum?id=qw8AKxfYbI
            uncond_mask = prob_mask_like((batch,), cond_dropout_prob, device = device)
            text_mask *= rearrange(~uncond_mask, 'b -> b 1')

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
