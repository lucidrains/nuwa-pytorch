from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from nuwa_pytorch.reversible import ReversibleSequence
from nuwa_pytorch.reversible_video_audio import DualModalityReversibleSequence

from unfoldNd import unfoldNd

from tqdm import tqdm

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

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

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

def batch_process(t, fn, chunks = 10, dim = 0):
    chunks = [fn(t_chunk) for t_chunk in t.chunk(chunks, dim = dim)]
    return torch.cat(chunks, dim = dim)

# gradient control

def frac_gradient(t, frac):
    return t * frac + t.detach() * (1 - frac)

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

# relative positional embedding (rotary)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# helper classes

class ShiftAudioTokens(nn.Module):
    def __init__(
        self,
        fn,
        audio_tokens_per_timestep = 1
    ):
        super().__init__()
        self.fn = fn
        self.audio_tokens_per_timestep = audio_tokens_per_timestep

    def forward(self, x, **kwargs):
        n = x.shape[1]

        # pad to nearest time step

        padding = self.audio_tokens_per_timestep - (n % self.audio_tokens_per_timestep)
        x = F.pad(x, (0, 0, 0, padding), value = 0.)

        # shift along time

        x_shift, x = x.chunk(2, dim = -1)
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
        x = torch.cat((x_shift, x), dim = -1)

        # remove padding if needed

        return self.fn(x[:, :n], **kwargs)

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

        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

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
        context_mask = None,
        rotary_pos_emb = None
    ):
        b, h, device = x.shape[0], self.heads, x.device

        has_context = exists(context)
        kv_input = context if has_context else x

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # add rotary positional embedding, if exists

        if not has_context and exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        # add null key / values, needed for condition dropout

        null_k = repeat(self.null_k, 'h 1 d -> b h 1 d', b = b)
        null_v = repeat(self.null_v, 'h 1 d -> b h 1 d', b = b)

        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

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

    def forward(self, x, **kwargs):
        b, n, _, h, device = *x.shape, self.heads, x.device

        # more variables

        dilation = self.dilation
        kernel_size = self.kernel_size
        video_padding = self.video_padding
        fmap_size = self.video_shape[1]

        bos_only = n == 1
        tokens_per_frame = fmap_size ** 2

        padding = padding_to_multiple_of(n - 1, tokens_per_frame)
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

            attn = stable_softmax(sim, dim = -1)

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

class SparseCausal2DNA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        height = 1,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        kernel_size = 5,
        dilation = 1
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.talking_heads = nn.Conv3d(heads, heads, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # handle variables for unfold

        self.height = height # width is the sequence length, time-axis - (batch, seq) -> (batch, time, height)

        self.kernel_size = (kernel_size, height)
        self.dilation = (dilation, 1)
        self.padding = (calc_same_padding(kernel_size, dilation), 0)

        # causal mask

        self.register_buffer('causal_mask', None, persistent = False)

    def get_causal_mask(self, t):
        if exists(self.causal_mask) and self.causal_mask.shape[-3] == t.shape[-3]:
            return self.causal_mask

        device, seq_len = t.device, t.shape[-3] * self.height
        q_range = torch.arange(seq_len, device = device, dtype = torch.float32)
        k_range = torch.arange(seq_len, device = device, dtype = torch.float32)

        q_range = rearrange(q_range, '(n m) -> n m', m = self.height)
        k_range = rearrange(k_range, '(n m) -> 1 1 n m', m = self.height)

        k_range = F.pad(k_range, (0, 0, self.padding[0], self.padding[0]), value = seq_len)
        k_range = unfoldNd(k_range, kernel_size = self.kernel_size, dilation = self.dilation)
        k_range = rearrange(k_range, '1 d n -> n d')

        causal_mask = rearrange(q_range, 'n i -> n i 1') < rearrange(k_range, 'n j -> n 1 j')
        causal_mask = F.pad(causal_mask, (1, 0), value = False)

        self.register_buffer('causal_mask', causal_mask, persistent = False)
        return causal_mask

    def forward(
        self,
        x,
        **kwargs
    ):
        b, n, h, device = x.shape[0], x.shape[1], self.heads, x.device

        tokens_per_timestep = self.height
        kernel_numel = self.kernel_size[0] * self.kernel_size[1]

        # pad to right length

        bos_only = n == 1
        seq_pad = padding_to_multiple_of(n - 1, tokens_per_timestep)

        # pad for last token in video

        padded_x = F.pad(x, (0, 0, 0, seq_pad), value = 0.) if seq_pad > 0 else x

        # derive queries, keys, values

        q, k, v = self.to_qkv(padded_x).chunk(3, dim = -1)

        # handle bos only

        if bos_only:
            return self.to_out(v)

        out_bos = v[:, :1]

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # handle bos

        (q_bos, q), (k_bos, k), (v_bos, v) = map(lambda t: (t[:, :, 0], t[:, :, 1:]), (q, k, v))

        # reshape key / values to be unfolded

        k, v = map(lambda t: rearrange(t, 'b h (x y) d -> (b h) d x y ', y = tokens_per_timestep), (k, v))
        k, v = map(lambda t: F.unfold(t, kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding), (k, v))
        k, v = map(lambda t: rearrange(t, '(b h f) (d j) i -> b h i (f j) d', b = b, h = h, j = kernel_numel), (k, v))

        # add bos

        k_bos_repeated, v_bos_repeated = map(lambda t: repeat(t, 'b h d -> b h i 1 d', i = k.shape[-3]), (k_bos, v_bos))
        k = torch.cat((k_bos_repeated, k), dim = -2)
        v = torch.cat((v_bos_repeated, v), dim = -2)

        q = rearrange(q, 'b h (x y) d -> b h x y d', y = tokens_per_timestep)

        sim = einsum('b h n i d, b h n j d -> b h n i j', q, k)

        # causal + padding mask

        mask_value = -torch.finfo(x.dtype).max
        causal_mask = self.get_causal_mask(sim)
        sim = sim.masked_fill(causal_mask, mask_value)

        # attention

        attn = stable_softmax(sim, dim = -1)
        attn = self.talking_heads(attn)
        attn = self.dropout(attn)

        # aggregate, merge, and combine heads

        out = einsum('b h n i j, b h n j d -> b h n i d', attn, v)
        out = rearrange(out, 'b h x y d -> b (x y) (h d)')

        # add output for bos back

        out = torch.cat((out_bos, out), dim = -2)

        return self.to_out(out[:, :n])

class SparseCross2DNA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        kernel_size = 3,
        dilation = 1,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.null_k = nn.Parameter(torch.randn(heads, 1, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, 1, dim_head))

        self.talking_heads = nn.Conv3d(heads, heads, 1, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # handle variables for 2d unfold

        self.image_size = image_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = calc_same_padding(kernel_size, dilation)

    def forward(
        self,
        x,
        *,
        context,
        context_mask = None,
        **kwargs
    ):
        b, n, h, device = x.shape[0], x.shape[1], self.heads, x.device

        fmap_size, kernel_size, dilation, padding = self.image_size, self.kernel_size, self.dilation, self.padding

        context_len = context.shape[-2]
        tokens_per_frame = fmap_size * fmap_size
        kernel_numel = kernel_size * kernel_size

        # always have context mask avaiable

        if not exists(context_mask):
            context_mask = torch.ones((b, context_len), dtype = torch.bool, device = device)

        mask_value = -torch.finfo(x.dtype).max

        # derive queries, keys, values

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # scale

        q = q * self.scale

        # handle bos

        q_bos, q = q[:, :, 0], q[:, :, 1:]

        null_k_for_bos = repeat(self.null_k, 'h 1 d -> b h 1 d', b = b)
        null_v_for_bos = repeat(self.null_v, 'h 1 d -> b h 1 d', b = b)

        k_for_bos = torch.cat((null_k_for_bos, k), dim = -2)
        v_for_bos = torch.cat((null_v_for_bos, v), dim = -2)

        sim_bos = einsum('b h d, b h j d -> b h j', q_bos, k_for_bos)

        bos_context_mask = rearrange(context_mask, 'b j -> b 1 j')
        bos_context_mask = F.pad(bos_context_mask, (1, 0), value = True)
        sim_bos = sim_bos.masked_fill(~bos_context_mask, mask_value)

        attn_bos = stable_softmax(sim_bos, dim = -1)
        out_bos = einsum('b h j, b h j d -> b h d', attn_bos, v_for_bos)
        out_bos = rearrange(out_bos, 'b h d -> b 1 (h d)')

        # early return if only bos token

        if n == 1:
            return self.to_out(out_bos)

        # reshape key / values to be unfolded

        k, v = map(lambda t: rearrange(t, 'b h (f x y) d -> (b h f) d x y', x = fmap_size, y = fmap_size), (k, v))
        k, v = map(lambda t: F.unfold(t, kernel_size = kernel_size, dilation = dilation, padding = padding), (k, v))
        k, v = map(lambda t: rearrange(t, '(b h f) (d j) i -> b h i (f j) d', b = b, h = h, j = kernel_numel), (k, v))

        # add null key / values, needed for condition dropout

        null_k = repeat(self.null_k, 'h 1 d -> b h i 1 d', b = b, i = tokens_per_frame)
        null_v = repeat(self.null_v, 'h 1 d -> b h i 1 d', b = b, i = tokens_per_frame)

        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        # pad queries to nearest frame

        q_padding = padding_to_multiple_of(q.shape[-2], tokens_per_frame)
        q = F.pad(q, (0, 0, 0, q_padding), value = 0.)

        # similarity

        q = rearrange(q, 'b h (f i) d -> b h f i d', i = tokens_per_frame)

        sim = einsum('b h f i d, b h i j d -> b h f i j', q, k)

        # masking

        context_mask = rearrange(context_mask, 'b (f x y) -> (b f) 1 x y', x = fmap_size, y = fmap_size)
        context_mask = F.unfold(context_mask.float(), kernel_size = kernel_size, dilation = dilation, padding = padding)
        context_mask = context_mask == 1.
        context_mask = rearrange(context_mask, '(b f) j i -> b 1 1 i (f j)', b = b, j = kernel_numel)
        context_mask = F.pad(context_mask, (1, 0), value = True) # always pay attention to null key / value

        sim = sim.masked_fill(~context_mask, mask_value)

        # attention

        attn = stable_softmax(sim, dim = -1)
        attn = self.talking_heads(attn)
        attn = self.dropout(attn)

        # aggregate, merge, and combine heads

        out = einsum('b h f i j, b h i j d -> b h f i d', attn, v)
        out = rearrange(out, 'b h f n d -> b (f n) (h d)')

        # add output for bos back

        out = torch.cat((out_bos, out), dim = 1)

        return self.to_out(out[:, :n])

"""
For efficient audio <-> video attention
Largely inspired by chunk cross attention from https://arxiv.org/abs/2112.04426
"""

class CrossModalityCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        chunk_size,
        context_chunk_size,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        has_start_token = True,
        context_has_start_token = True,
        norm = False,
        norm_context = False,
        dropout = 0.
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim  = dim_head * heads

        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.null_k = nn.Parameter(torch.randn(heads, dim_head))
        self.null_v = nn.Parameter(torch.randn(heads, dim_head))

        self.talking_heads = nn.Conv3d(heads, heads, 1)
        self.dropout = nn.Dropout(dropout)

        self.has_start_token = has_start_token
        self.context_has_start_token = context_has_start_token

        self.chunk_size = chunk_size
        self.context_chunk_size = context_chunk_size

    def forward(
        self,
        seq,
        context,
        mask = None,
        context_mask = None
    ):
        seq_shape, device = seq.shape, seq.device

        # get lengths of sequence and context, excluding start token

        seq_len = seq.shape[-2] - (1 if self.has_start_token else 0)
        context_len = context.shape[-2] - (1 if self.context_has_start_token else 0)

        # determine padding
        # depending on whether start token exists

        seq_left_pad = -1 if self.has_start_token else 0
        seq_right_pad = padding_to_multiple_of(seq_len, self.chunk_size)

        seq_out_left_pad = -seq_left_pad
        seq_out_right_pad = -seq_right_pad

        context_left_pad = self.context_chunk_size - (1 if self.context_chunk_size else 0)
        context_right_pad = padding_to_multiple_of(context_len, self.context_chunk_size)

        # do actual padding so divisible by chunk size (video frame)

        seq = F.pad(seq, (0, 0, seq_left_pad, seq_right_pad), value = 0.)
        context = F.pad(context, (0, 0, context_left_pad, context_right_pad), value = 0.)

        if exists(context_mask):
            context_mask = F.pad(context_mask, (context_left_pad, context_right_pad), value = False)

        # break into chunks

        """
        b - batch
        n - num chunks
        c - chunks
        d - feature dimension
        h - heads
        """

        seq = rearrange(seq, 'b (n c) d -> b n c d', c = self.chunk_size)
        context = rearrange(context, 'b (n c) d -> b n c d', c = self.context_chunk_size)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b (n c) -> b n c', c = self.context_chunk_size)

        # determine if sequence is longer than context, or vice versa, when aligned for time

        seq_num_chunks = seq.shape[-3]
        context_num_chunks = context.shape[-3]

        if seq_num_chunks <= context_num_chunks:
            context = context[:, :seq_num_chunks]

            if exists(context_mask):
                context_mask = context_mask[:, :seq_num_chunks]
        else:
            # handle the case where the sequence has more chunks
            # in which case the sequence is curtailed, and output of attention is 0 for the excised right portion

            seq = seq[:, :context_num_chunks]
            seq_out_right_pad += self.chunk_size * (seq_num_chunks - context_num_chunks)

        # early exit if nothing to attend to

        if context.shape[1] == 0:
            return torch.zeros(seq_shape, device = device)

        # pre layernorm

        seq = self.norm(seq)
        context = self.context_norm(context)

        # attention time!

        q = self.to_q(seq)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n c (h d) -> b h n c d', h = self.heads), (q, k, v))
        q = q * self.scale

        null_k, null_v = map(lambda t: repeat(t, 'h d -> b h n 1 d', b = q.shape[0], n = q.shape[2]), (self.null_k, self.null_v))

        k = torch.cat((null_k, k), dim = -2)
        v = torch.cat((null_v, v), dim = -2)

        sim = einsum('b h n i d, b h n j d -> b h n i j', q, k)

        if exists(context_mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            context_mask = rearrange(context_mask, 'b n c -> b 1 n 1 c')
            context_mask = F.pad(context_mask, (1, 0), value = True) # null key / value
            sim = sim.masked_fill(~context_mask, max_neg_value)

        attn = stable_softmax(sim, dim = -1)
        attn = self.dropout(attn)

        attn = self.talking_heads(attn)

        out = einsum('b h n i j, b h n j d -> b h n i d', attn, v)
        out = rearrange(out, 'b h n c d -> b (n c) (h d)')
        out = self.to_out(out)

        # shift back to original sequence

        out = F.pad(out, (0, 0, seq_out_left_pad, seq_out_right_pad), value = 0.)

        # mask src sequence, if mask was passed in (extra insurance)

        if exists(mask):
            mask = rearrange(mask, '... -> ... 1')
            out = out.masked_fill(~mask, 0.)

        return out

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
        cross_2dna_attn = False,
        cross_2dna_image_size = None,
        cross_2dna_kernel_size = 3,
        cross_2dna_dilations = (1,),
        sparse_3dna_attn = False,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_video_shape = None,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        shift_video_tokens = False,
        rotary_pos_emb = False
    ):
        super().__init__()
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'
        assert not (cross_2dna_attn and not exists(cross_2dna_image_size)), 'cross_2dna_image_size must be defined'

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

            cross_attn = None

            if cross_attend:
                if cross_2dna_attn:
                    dilation = cross_2dna_dilations[ind % len(cross_2dna_dilations)]

                    cross_attn = SparseCross2DNA(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        dropout = attn_dropout,
                        image_size = cross_2dna_image_size,
                        kernel_size = cross_2dna_kernel_size,
                        dilation = dilation
                    )

                else:
                    cross_attn = Attention(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        dropout = attn_dropout
                    )

            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            if sparse_3dna_attn and shift_video_tokens:
                fmap_size = sparse_3dna_video_shape[-1]
                self_attn = ShiftVideoTokens(self_attn, image_size = fmap_size)
                ff        = ShiftVideoTokens(ff, image_size = fmap_size)

            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = self_attn),
                SandwichNorm(dim = dim, fn = cross_attn) if cross_attend else None,
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
        cross_2dna_attn = False,
        cross_2dna_image_size = None,
        cross_2dna_kernel_size = 3,
        cross_2dna_dilations = (1,),
        sparse_3dna_attn = False,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_video_shape = None,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        shift_video_tokens = False,
        rotary_pos_emb = False
    ):
        super().__init__()
        assert not (sparse_3dna_attn and not exists(sparse_3dna_video_shape)), 'sparse_3dna_video_shape must be defined if turned on'
        assert not (cross_2dna_attn and not exists(cross_2dna_image_size)), 'cross_2dna_image_size must be defined'

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

            if cross_2dna_attn:
                dilation = cross_2dna_dilations[ind % len(cross_2dna_dilations)]

                cross_attn = SparseCross2DNA(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = attn_dropout,
                    image_size = cross_2dna_image_size,
                    kernel_size = cross_2dna_kernel_size,
                    dilation = dilation
                )
            else:
                cross_attn = Attention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    dropout = attn_dropout
                )

            self.layers.append(MList([
                SandwichNorm(dim = dim, fn = cross_attn),
                SandwichNorm(dim = dim, fn = wrapper_fn(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)))
            ]))

        attn_context_layer = ((True, False),) if cross_attend else tuple()
        route_attn = ((True, False), *attn_context_layer) * depth
        route_context = ((False, False), *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        attn_route_map = {'mask': route_attn, 'rotary_pos_emb': route_attn}

        self.net = ReversibleSequence(self.layers, args_route = {**context_route_map, **attn_route_map})
        self.norm = StableLayerNorm(dim)

    def forward(
        self,
        x,
        **kwargs
    ):
        x = self.net(x, **kwargs)
        return self.norm(x)

# dual modality decoder (for video and audio synthesis)

class DualModalityDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_audio_tokens_per_video_frame,
        num_video_tokens_per_frame,
        sparse_3dna_video_shape,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        sparse_2dna_kernel_size = 7,
        sparse_2dna_dilation = (1,),
        shift_video_tokens = False,
        shift_audio_tokens = False,
        audio_tokens_per_timestep = 1,
        cross_modality_attn_every = 3
    ):
        super().__init__()
        self.layers = MList([])
        self.layer_types = []

        def video_intra_modality_attn():
            dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]

            self_attn = Sparse3DNA(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                causal = True,
                kernel_size = sparse_3dna_kernel_size,
                dilation = dilation,
                video_shape = sparse_3dna_video_shape,
                query_num_frames_chunk = sparse_3dna_query_num_frames_chunk
            )

            cross_attn = Attention(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                dropout = attn_dropout
            )

            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            if shift_video_tokens:
                fmap_size = sparse_3dna_video_shape[-1]
                self_attn = ShiftVideoTokens(self_attn, image_size = fmap_size)
                ff        = ShiftVideoTokens(ff, image_size = fmap_size)

            return MList([
                SandwichNorm(dim = dim, fn = self_attn),
                SandwichNorm(dim = dim, fn = cross_attn),
                SandwichNorm(dim = dim, fn = ff)
            ])

        def audio_intra_modality_attn():
            dilation = sparse_2dna_dilation[ind % len(sparse_2dna_dilation)]

            self_attn = SparseCausal2DNA(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                kernel_size = sparse_2dna_kernel_size,
                dilation = dilation,
                dropout = attn_dropout
            )

            cross_attn = Attention(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                dropout = attn_dropout
            )

            ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            if shift_audio_tokens:
                self_attn = ShiftAudioTokens(self_attn, audio_tokens_per_timestep = audio_tokens_per_timestep)
                ff        = ShiftAudioTokens(ff, audio_tokens_per_timestep = audio_tokens_per_timestep)

            return MList([
                SandwichNorm(dim = dim, fn = self_attn),
                SandwichNorm(dim = dim, fn = cross_attn),
                SandwichNorm(dim = dim, fn = ff)
            ])

        def inter_modality_attn(chunk_size, context_chunk_size):
            cross_modality_attn = CrossModalityCrossAttention(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                chunk_size = chunk_size,
                context_chunk_size = context_chunk_size,
                has_start_token = True,
                context_has_start_token = True
            )

            cross_modality_ff = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

            return MList([
                SandwichNorm(dim = dim, fn = cross_modality_attn),
                SandwichNorm(dim = dim, fn = cross_modality_ff),
            ])

        for ind in range(depth):
            video_modality_attn = video_intra_modality_attn()
            audio_modality_attn = audio_intra_modality_attn()

            self.layer_types.append('intra_modality')

            self.layers.append(MList([
                video_modality_attn,
                audio_modality_attn,
            ]))

            if ((ind + 1) % cross_modality_attn_every) == 0:
                self.layer_types.append('inter_modality')

                video_to_audio_attn = inter_modality_attn(num_video_tokens_per_frame, num_audio_tokens_per_video_frame)
                audio_to_video_attn = inter_modality_attn(num_audio_tokens_per_video_frame, num_video_tokens_per_frame)

                self.layers.append(MList([
                    video_to_audio_attn,
                    audio_to_video_attn
                ]))

        self.video_norm = StableLayerNorm(dim)
        self.audio_norm = StableLayerNorm(dim)

    def forward(
        self,
        video,
        audio,
        *,
        context,
        audio_mask = None,
        video_mask = None,
        context_mask = None,
        **kwargs
    ):
        for blocks, layer_type in zip(self.layers, self.layer_types):
            if layer_type == 'intra_modality':
                (video_self_attn, video_cross_attn, video_ff), (audio_self_attn, audio_cross_attn, audio_ff) = blocks

                video_ = video_self_attn(video, mask = video_mask) + video
                video_ = video_cross_attn(video_, context = context, mask = video_mask, context_mask = context_mask) + video_
                video_ = video_ff(video_) + video_

                audio_ = audio_self_attn(audio, mask = audio_mask) + audio
                audio_ = audio_cross_attn(audio_, context = context, mask = audio_mask, context_mask = context_mask) + audio_
                audio_ = audio_ff(audio_) + audio_

            elif layer_type == 'inter_modality':
                (video_to_audio_attn, video_ff), (audio_to_video_attn, audio_ff) = blocks

                video_ = video_to_audio_attn(
                    video,
                    context = audio,
                    mask = video_mask,
                    context_mask = audio_mask
                ) + video

                audio_ = audio_to_video_attn(
                    audio,
                    context = video,
                    mask = audio_mask,
                    context_mask = video_mask
                ) + audio

                video_ = video_ff(video_) + video_
                audio_ = audio_ff(audio_) + audio_
            else:
                raise ValueError(f'unknown layer type {layer_type}')

            video, audio = video_, audio_

        return self.video_norm(video), self.audio_norm(audio)

class ReversibleDualModalityDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_audio_tokens_per_video_frame,
        num_video_tokens_per_frame,
        sparse_3dna_video_shape,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilations = (1,),
        sparse_2dna_kernel_size = 7,
        sparse_2dna_dilation = (1,),
        shift_video_tokens = False,
        shift_audio_tokens = False,
        audio_tokens_per_timestep = 1,
        cross_modality_attn_every = 3
    ):
        super().__init__()
        self.layers = MList([])
        self.layer_types = []

        norm_wrapper = lambda fn: SandwichNorm(dim = dim, fn = fn)
        create_ff = lambda: FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, chunk_size = ff_chunk_size)

        for ind in range(depth):
            video_dilation = sparse_3dna_dilations[ind % len(sparse_3dna_dilations)]
            audio_dilation = sparse_2dna_dilation[ind % len(sparse_2dna_dilation)]

            video_self_attn = Sparse3DNA(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                causal = True,
                kernel_size = sparse_3dna_kernel_size,
                dilation = video_dilation,
                video_shape = sparse_3dna_video_shape,
                query_num_frames_chunk = sparse_3dna_query_num_frames_chunk
            )

            audio_self_attn = SparseCausal2DNA(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                dropout = attn_dropout,
                kernel_size = sparse_2dna_kernel_size,
                dilation = audio_dilation
            )

            video_ff = create_ff()
            audio_ff = create_ff()

            if shift_video_tokens:
                fmap_size = sparse_3dna_video_shape[-1]
                video_self_attn = ShiftVideoTokens(video_self_attn, image_size = fmap_size)
                video_ff        = ShiftVideoTokens(video_ff, image_size = fmap_size)

            if shift_audio_tokens:
                audio_self_attn = ShiftAudioTokens(audio_self_attn, audio_tokens_per_timestep = audio_tokens_per_timestep)
                audio_ff        = ShiftAudioTokens(audio_ff, audio_tokens_per_timestep = audio_tokens_per_timestep)

            self.layers.append(MList([
                norm_wrapper(video_self_attn),
                norm_wrapper(video_ff),
                norm_wrapper(audio_self_attn),
                norm_wrapper(audio_ff)
            ]))

            self.layer_types.append('intra_modality_self_attn')

            video_cross_attn = Attention(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                dropout = attn_dropout
            )

            audio_cross_attn = Attention(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                dropout = attn_dropout
            )

            video_cross_ff = create_ff()
            audio_cross_ff = create_ff()

            self.layers.append(MList([
                norm_wrapper(video_cross_attn),
                norm_wrapper(video_cross_ff),
                norm_wrapper(audio_cross_attn),
                norm_wrapper(audio_cross_ff)
            ]))

            self.layer_types.append('intra_modality_cross_attn')

            if ((ind + 1) % cross_modality_attn_every) == 0:
                video_to_audio_attn = CrossModalityCrossAttention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    chunk_size = num_video_tokens_per_frame,
                    context_chunk_size = num_audio_tokens_per_video_frame,
                    has_start_token = True,
                    context_has_start_token = True
                )

                video_cross_modality_ff = create_ff()

                audio_to_video_attn = CrossModalityCrossAttention(
                    dim = dim,
                    heads = heads,
                    dim_head = dim_head,
                    chunk_size = num_audio_tokens_per_video_frame,
                    context_chunk_size = num_video_tokens_per_frame,
                    has_start_token = True,
                    context_has_start_token = True
                )

                audio_cross_modality_ff = create_ff()

                self.layers.append(MList([
                    video_to_audio_attn,
                    video_cross_modality_ff,
                    audio_to_video_attn,
                    audio_cross_modality_ff
                ]))

                self.layer_types.append('inter_modality_cross_attn')

        self.net = DualModalityReversibleSequence(self.layers, self.layer_types)

        self.video_norm = StableLayerNorm(dim)
        self.audio_norm = StableLayerNorm(dim)

    def forward(
        self,
        video,
        audio,
        *,
        context,
        audio_mask = None,
        video_mask = None,
        context_mask = None,
        **kwargs
    ):
        video, audio = self.net(
            video,
            audio,
            context = context,
            audio_mask = audio_mask,
            video_mask = video_mask,
            context_mask = context_mask
        )

        return self.video_norm(video), self.audio_norm(audio)

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
        dim,
        vae = None,
        image_size = None,
        max_video_frames = 5,
        text_num_tokens = 49408,
        text_max_seq_len = 256,
        text_enc_depth = 6,
        text_enc_dim_head = 64,
        text_enc_heads = 8,
        text_rotary_pos_emb = True,
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
        assert exists(vae) ^ exists(image_size), 'either VAE or image size must be specified'

        self.vae = None
        if exists(vae):
            self.vae = vae.copy_for_eval()
            image_size = vae.image_size

        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = Embedding(text_num_tokens, dim, frac_gradient = embed_gradient_frac)

        # positional embedding for text

        self.text_abs_pos_emb = Embedding(text_max_seq_len, dim)  if not text_rotary_pos_emb else None
        self.text_rotary_pos_emb = RotaryEmbedding(dim = min(32, text_enc_dim_head)) if text_rotary_pos_emb else None

        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        self.text_transformer = enc_transformer_klass(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            rotary_pos_emb = text_rotary_pos_emb
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

        dec_transformer_klass = Transformer if not dec_reversible else ReversibleTransformer

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
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_attn = True,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk
        )

        self.to_logits = nn.Linear(dim, num_image_tokens)

    def embed_text(self, text, mask = None):
        batch, seq_len, device = *text.shape, text.device
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        tokens = self.text_embedding(text)

        if exists(self.text_abs_pos_emb):
            pos_emb = self.text_abs_pos_emb(torch.arange(seq_len, device = device))
            tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.text_rotary_pos_emb):
            rotary_pos_emb = self.text_rotary_pos_emb(seq_len, device = device)

        return self.text_transformer(
            tokens,
            mask = mask,
            rotary_pos_emb = rotary_pos_emb
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

        for ind in tqdm(range(total_video_tokens)):
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
        video = None,
        return_loss = False,
        cond_dropout_prob = 0.2
    ):
        batch, seq_len, frames, device = *text.shape, video.shape[1], text.device

        text_mask = text != 0
        text_embeds = self.embed_text(text, mask = text_mask)

        if video.dtype == torch.long:
            frame_indices = video
        else:
            assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'
            assert exists(self.vae), 'VAE must be passed in if you wish for video to be encoded to ids automatically'
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

# generating video and audio

class NUWAVideoAudio(nn.Module):
    def __init__(
        self,
        *,
        vae,
        dim,
        image_size,
        num_audio_tokens,
        num_audio_tokens_per_video_frame,
        audio_tokens_per_timestep = 1,
        max_video_frames = 5,
        text_num_tokens = 49408,
        text_max_seq_len = 256,
        text_enc_depth = 6,
        text_enc_dim_head = 64,
        text_enc_heads = 8,
        text_rotary_pos_emb = False,
        enc_reversible = False,
        dec_reversible = True,
        dec_depth = 6,
        dec_dim_head = 64,
        dec_heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_chunk_size = None,
        embed_gradient_frac = 0.2,
        shift_video_tokens = True,
        shift_audio_tokens = True,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_query_num_frames_chunk = None,
        sparse_3dna_dilation = 1,
        sparse_2dna_kernel_size = 7,
        sparse_2dna_dilation = 1,
        audio_loss_weight = 1.,
        cross_modality_attn_every = 3
    ):
        super().__init__()
        self.vae = vae.copy_for_eval()
        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.text_max_seq_len = text_max_seq_len
        self.text_embedding = Embedding(text_num_tokens, dim, frac_gradient = embed_gradient_frac)

        self.text_abs_pos_emb = Embedding(text_max_seq_len, dim) if not text_rotary_pos_emb else None
        self.text_rotary_pos_emb = RotaryEmbedding(dim = min(32, text_enc_dim_head)) if text_rotary_pos_emb else None

        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        self.text_transformer = enc_transformer_klass(
            dim = dim,
            depth = text_enc_depth,
            heads = text_enc_heads,
            dim_head = text_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # video related params

        self.video_bos = nn.Parameter(torch.randn(dim))
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        fmap_size = image_size // (2 ** vae_num_layers)

        self.video_fmap_size = fmap_size
        self.max_video_frames = max_video_frames
        video_shape = (max_video_frames, fmap_size, fmap_size)

        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # audio related params

        self.audio_bos = nn.Parameter(torch.randn(dim))
        self.audio_embedding = Embedding(num_audio_tokens, dim, frac_gradient = embed_gradient_frac)

        max_audio_seq_len = num_audio_tokens_per_video_frame * max_video_frames
        self.audio_pos_emb = nn.Embedding(max_audio_seq_len, dim)

        self.audio_loss_weight = audio_loss_weight

        # num tokens per video frame

        self.num_video_tokens_per_frame = fmap_size ** 2
        self.num_audio_tokens_per_video_frame = num_audio_tokens_per_video_frame

        # cycle dilation for sparse 3d-nearby attention

        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        sparse_2dna_dilation = tuple(range(1, sparse_2dna_dilation + 1)) if not isinstance(sparse_2dna_dilation, (list, tuple)) else sparse_2dna_dilation

        decoder_klass = ReversibleDualModalityDecoder if dec_reversible else DualModalityDecoder

        self.video_audio_transformer = decoder_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            audio_tokens_per_timestep = audio_tokens_per_timestep,
            shift_audio_tokens = shift_audio_tokens,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_2dna_kernel_size = sparse_2dna_kernel_size,
            sparse_2dna_dilation = sparse_2dna_dilation,
            num_audio_tokens_per_video_frame = num_audio_tokens_per_video_frame,
            num_video_tokens_per_frame = fmap_size * fmap_size,
            cross_modality_attn_every = cross_modality_attn_every
        )

        self.to_video_logits = nn.Linear(dim, num_image_tokens)
        self.to_audio_logits = nn.Linear(dim, num_audio_tokens)

    def embed_text(self, text, mask = None):
        batch, seq_len, device = *text.shape, text.device
        assert seq_len <= self.text_max_seq_len, 'your input text has a greater length than what was designated on initialization'

        tokens = self.text_embedding(text)

        if exists(self.text_abs_pos_emb):
            pos_emb = self.text_abs_pos_emb(torch.arange(seq_len, device = device))
            tokens = tokens + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.text_rotary_pos_emb):
            rotary_pos_emb = self.text_rotary_pos_emb(seq_len, device = device)

        return self.text_transformer(
            tokens,
            mask = mask,
            rotary_pos_emb = rotary_pos_emb
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
        num_tokens_per_frame, num_audio_tokens_per_video_frame = self.num_video_tokens_per_frame, self.num_audio_tokens_per_video_frame

        text_mask = text != 0
        text_embeds = self.embed_text(text, mask = text_mask)

        video_bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        audio_bos = repeat(self.audio_bos, 'd -> b 1 d', b = batch)

        video_indices = torch.empty((batch, 0), device = device, dtype = torch.long)
        audio_indices = torch.empty((batch, 0), device = device, dtype = torch.long)

        num_frames = default(num_frames, self.max_video_frames)

        total_video_tokens = num_frames * num_tokens_per_frame
        total_audio_tokens = num_frames * num_audio_tokens_per_video_frame

        video_pos_emb = self.video_pos_emb()

        is_decoding_video = True # toggle to False to decode audio, alternating between video and audio

        while video_indices.shape[1] < total_video_tokens \
            or audio_indices.shape[1] < total_audio_tokens:

            video_indices_input = video_indices
            audio_indices_input = audio_indices

            num_video_tokens = video_indices.shape[1]
            if num_video_tokens > total_video_tokens:
                curr_frame_tokens = num_video_tokens % num_tokens_per_frame
                lookback_tokens = (self.max_video_frames - (0 if curr_frame_tokens == 0 else 1)) * num_tokens_per_frame + curr_frame_tokens
                video_indices_input = video_indices[:, -lookback_tokens:]

            # prep video embeddings

            frame_embeddings = self.image_embedding(video_indices_input)
            frame_embeddings = video_pos_emb[:, :frame_embeddings.shape[1]] + frame_embeddings
            frame_embeddings = torch.cat((video_bos, frame_embeddings), dim = 1)

            # prep audio embeddings

            audio_embeddings = self.audio_embedding(audio_indices_input)
            audio_pos_emb = self.audio_pos_emb(torch.arange(audio_embeddings.shape[1], device = device))
            audio_pos_emb = rearrange(audio_pos_emb, 'n d -> 1 n d')
            audio_embeddings = audio_embeddings + audio_pos_emb
            audio_embeddings = torch.cat((audio_bos, audio_embeddings), dim = 1)

            frame_embeddings, audio_embeddings = self.video_audio_transformer(
                frame_embeddings,
                audio_embeddings,
                context = text_embeds,
                context_mask = text_mask
            )

            logits = self.to_video_logits(frame_embeddings) if is_decoding_video else self.to_audio_logits(audio_embeddings)

            if cond_scale != 1:
                # discovery by Katherine Crowson
                # https://twitter.com/RiversHaveWings/status/1478093658716966912
                uncond_frame_embeddings, uncond_audio_embeddings = self.video_audio_transformer(
                    frame_embeddings,
                    audio_embeddings,
                    context = text_embeds,
                    context_mask = torch.zeros_like(text_mask).bool()
                )

                uncond_logits = self.to_video_logits(uncond_frame_embeddings) if is_decoding_video else self.to_audio_logits(uncond_audio_embeddings)
                logits = uncond_logits + (logits - uncond_logits) * cond_scale

            logits = logits[:, -1, :]

            filtered_logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sample = rearrange(sample, 'b -> b 1')

            if is_decoding_video:
                video_indices = torch.cat((video_indices, sample), dim = 1)
                at_frame_boundary = (video_indices.shape[1] % num_tokens_per_frame) == 0
            else:
                audio_indices = torch.cat((audio_indices, sample), dim = 1)
                at_frame_boundary = (audio_indices.shape[1] % num_audio_tokens_per_video_frame) == 0

            # alternate between audio and video decoding, one video frame at a time

            if at_frame_boundary:
                is_decoding_video = not is_decoding_video

        # decoding video codebook indices with VQGan

        codes = self.vae.codebook[video_indices]
        codes = rearrange(codes, 'b (f h w) d -> (b f) d h w', h = self.video_fmap_size, w = self.video_fmap_size)

        image_reconstructions = batch_process(codes, self.vae.decode, chunks = decode_max_batchsize)
        video = rearrange(image_reconstructions, '(b f) d h w -> b f d h w', b = batch)

        # just return audio token indices for now

        audio = audio_indices

        return video, audio

    def forward(
        self,
        *,
        text,
        video,
        audio,
        return_loss = False,
        cond_dropout_prob = 0.2
    ):
        batch, seq_len, frames, device = *text.shape, video.shape[1], text.device

        text_mask = text != 0
        text_embeds = self.embed_text(text, mask = text_mask)

        # prep video representation

        if video.dtype == torch.long:
            frame_indices = video
        else:
            assert frames == self.max_video_frames, f'you must give the full video frames ({self.max_video_frames}) during training'
            assert exists(self.vae), 'VAE must be passed in if you wish for video to be encoded to ids automatically'
            frame_indices = self.vae.get_video_indices(video)

        frame_indices = rearrange(frame_indices, 'b ... -> b (...)')
        frame_indices_input = frame_indices[:, :-1] if return_loss else frame_indices

        frame_embeddings = self.image_embedding(frame_indices_input)
        frame_embeddings = self.video_pos_emb()[:, :-1] + frame_embeddings

        video_bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)
        frame_embeddings = torch.cat((video_bos, frame_embeddings), dim = 1)

        # prep audio representations

        audio_indices_input = audio[:, :-1] if return_loss else audio

        audio_embeddings = self.audio_embedding(audio_indices_input)
        audio_pos_emb = self.audio_pos_emb(torch.arange(audio_embeddings.shape[1], device = device))
        audio_embeddings = audio_embeddings + rearrange(audio_pos_emb, 'n d -> 1 n d')

        audio_bos = repeat(self.audio_bos, 'd -> b 1 d', b = batch)
        audio_embeddings = torch.cat((audio_bos, audio_embeddings), dim = 1)

        # null conditions, for super-conditioning

        if self.training and cond_dropout_prob > 0:
            # dropout condition randomly
            # presented in https://openreview.net/forum?id=qw8AKxfYbI
            uncond_mask = prob_mask_like((batch,), cond_dropout_prob, device = device)
            text_mask *= rearrange(~uncond_mask, 'b -> b 1')

        # twin attention towers for video and audio, with efficient chunked cross modality attention

        frame_embeddings, audio_embeddings = self.video_audio_transformer(
            frame_embeddings,
            audio_embeddings,
            context = text_embeds,
            context_mask = text_mask
        )

        video_logits = self.to_video_logits(frame_embeddings)
        audio_logits = self.to_audio_logits(audio_embeddings)

        if not return_loss:
            return video_logits, audio_logits

        video_loss = F.cross_entropy(rearrange(video_logits, 'b n c -> b c n'), frame_indices)
        audio_loss = F.cross_entropy(rearrange(audio_logits, 'b n c -> b c n'), audio)

        return video_loss + audio_loss * self.audio_loss_weight

# main class for learning on sketches

class NUWASketch(nn.Module):
    def __init__(
        self,
        *,
        vae,
        sketch_vae,
        dim,
        image_size,
        max_video_frames = 5,
        sketch_max_video_frames = 2,
        sketch_enc_depth = 6,
        sketch_enc_dim_head = 64,
        sketch_enc_heads = 8,
        sketch_enc_use_sparse_3dna = False,
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
        cross_2dna_kernel_size = 3,
        cross_2dna_dilation = 1,
        sparse_3dna_kernel_size = 3,
        sparse_3dna_dilation = 1,
        sparse_3dna_query_num_frames_chunk = None,
    ):
        super().__init__()
        self.image_size = image_size

        self.sketch_vae = sketch_vae
        sketch_vae_num_layers = sketch_vae.num_layers
        sketch_num_image_tokens = sketch_vae.codebook_size
        sketch_fmap_size = image_size // (2 ** sketch_vae_num_layers)

        sketch_shape = (sketch_max_video_frames, sketch_fmap_size, sketch_fmap_size)

        self.sketch_max_video_frames = sketch_max_video_frames
        self.sketch_embedding = Embedding(sketch_num_image_tokens, dim, frac_gradient = embed_gradient_frac)
        self.sketch_pos_emb = AxialPositionalEmbedding(dim, shape = sketch_shape)

        # sparse 3dna kwargs

        sparse_3dna_dilations = tuple(range(1, sparse_3dna_dilation + 1)) if not isinstance(sparse_3dna_dilation, (list, tuple)) else sparse_3dna_dilation

        # encoder

        enc_transformer_klass = Transformer if not enc_reversible else ReversibleTransformer

        self.sketch_transformer = enc_transformer_klass(
            dim = dim,
            depth = sketch_enc_depth,
            heads = sketch_enc_heads,
            dim_head = sketch_enc_dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = sketch_shape,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_attn = sketch_enc_use_sparse_3dna
        )

        # decoder parameters

        self.vae = vae.copy_for_eval()

        vae_num_layers = vae.num_layers
        num_image_tokens = vae.codebook_size

        self.video_bos = nn.Parameter(torch.randn(dim))
        self.image_embedding = Embedding(num_image_tokens, dim, frac_gradient = embed_gradient_frac)

        fmap_size = image_size // (2 ** vae_num_layers)

        assert fmap_size == sketch_fmap_size, 'feature map size of video must be equal to the feature map size of sketches (VAEs must have same number of layers)'

        self.video_fmap_size = fmap_size
        self.max_video_frames = max_video_frames
        video_shape = (max_video_frames, fmap_size, fmap_size)

        self.video_pos_emb = AxialPositionalEmbedding(dim, shape = video_shape)

        # cycle dilation for sparse 3d-nearby attention

        cross_2dna_dilations = tuple(range(1, cross_2dna_dilation + 1)) if not isinstance(cross_2dna_dilation, (list, tuple)) else cross_2dna_dilation
        dec_transformer_klass = Transformer if not dec_reversible else ReversibleTransformer

        self.video_transformer = dec_transformer_klass(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            dim_head = dec_dim_head,
            causal = True,
            cross_attend = True,
            cross_2dna_attn = True,
            cross_2dna_image_size = fmap_size,
            cross_2dna_kernel_size = cross_2dna_kernel_size,
            cross_2dna_dilations = cross_2dna_dilations,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_chunk_size = ff_chunk_size,
            shift_video_tokens = shift_video_tokens,
            sparse_3dna_video_shape = video_shape,
            sparse_3dna_kernel_size = sparse_3dna_kernel_size,
            sparse_3dna_dilations = sparse_3dna_dilations,
            sparse_3dna_query_num_frames_chunk = sparse_3dna_query_num_frames_chunk,
            sparse_3dna_attn = True
        )

        self.to_logits = nn.Linear(dim, num_image_tokens)

    def embed_sketch(self, sketch, mask = None):
        batch, frames, channels, image_size, _, device = *sketch.shape, sketch.device

        if exists(mask):
            assert mask.shape[:2] == (batch, frames), 'sketch mask must be in shape of (batch x frame)'

        sketch_indices = self.sketch_vae.get_video_indices(sketch)
        sketch_indices = rearrange(sketch_indices, 'b ... -> b (...)')
        sketch_tokens = self.sketch_embedding(sketch_indices)

        num_tokens = sketch_tokens.shape[1]

        sketch_pos_emb = self.sketch_pos_emb()
        sketch_pos_emb = sketch_pos_emb[:, :num_tokens]

        sketch_tokens = sketch_tokens + sketch_pos_emb

        if exists(mask):
            mask = repeat(mask, 'b f -> b (f n)', n = (num_tokens // frames))
        else:
            mask = torch.ones((batch, num_tokens), dtype = torch.bool, device = device)

        embed = self.sketch_transformer(sketch_tokens, mask = mask)
        return embed, mask

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        sketch,
        sketch_mask = None,
        filter_thres = 0.9,
        temperature = 1.,
        decode_max_batchsize = 10,
        cond_scale = 2.,
        num_frames = None
    ):
        batch, device = sketch.shape[0], sketch.device

        sketch_embeds, decoder_context_mask = self.embed_sketch(sketch, mask = sketch_mask)

        bos = repeat(self.video_bos, 'd -> b 1 d', b = batch)

        video_indices = torch.empty((batch, 0), device = device, dtype = torch.long)

        num_tokens_per_frame = self.video_fmap_size ** 2

        num_frames = default(num_frames, self.max_video_frames)
        total_video_tokens =  num_tokens_per_frame * num_frames
        max_video_tokens = num_tokens_per_frame * self.max_video_frames

        pos_emb = self.video_pos_emb()

        for ind in tqdm(range(total_video_tokens)):
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
                context = sketch_embeds,
                context_mask = decoder_context_mask
            )

            logits = self.to_logits(frame_embeddings)

            if cond_scale != 1:
                # discovery by Katherine Crowson
                # https://twitter.com/RiversHaveWings/status/1478093658716966912
                uncond_frame_embeddings = self.video_transformer(
                    frame_embeddings,
                    context = sketch_embeds,
                    context_mask = torch.zeros_like(decoder_context_mask).bool()
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
        sketch,
        sketch_mask = None,
        video = None,
        return_loss = False,
        cond_dropout_prob = 0.2
    ):
        # handle one sketch gracefully

        if sketch.ndim == 4:
            sketch = rearrange(sketch, 'b c h w -> b 1 c h w')

        # get a bunch of variables

        batch, sketch_frames, sketch_channels, sketch_image_size, _, frames, device = *sketch.shape, video.shape[1], sketch.device

        # guardrails

        assert sketch_image_size == self.image_size, 'sketch image size must be equal'
        assert sketch_frames <= self.sketch_max_video_frames, 'sketch frames must be less than max sketch video frames'

        # get sketch embeddings, and calculate mask (for now, assume no padding)

        sketch_embeds, decoder_context_mask = self.embed_sketch(sketch, mask = sketch_mask)

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
            sketch_mask *= rearrange(~uncond_mask, 'b -> b 1')

        frame_embeddings = self.video_transformer(
            frame_embeddings,
            context = sketch_embeds,
            context_mask = decoder_context_mask
        )

        logits = self.to_logits(frame_embeddings)

        if not return_loss:
            return logits

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), frame_indices)
        return loss
