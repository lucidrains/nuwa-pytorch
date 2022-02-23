import torch
import torch.nn as nn
from torch.autograd.function import Function
from contextlib import contextmanager

from nuwa_pytorch.reversible import Deterministic

from einops import reduce

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# reversible self attention block

class ReversibleSelfAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)        

    def forward(self, x, m, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        fn_context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with fn_context():
            y1 = x1 + self.f(x2, record_rng = record_rng)
            y2 = x2 + self.g(y1, record_rng = record_rng)
            n1 = m1 + self.j(m2, record_rng = record_rng)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, **kwargs):
        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng = True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng = True)
            torch.autograd.backward(fx2, dx1, retain_graph = True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.k(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            fm2 = self.j(m2, set_rng = True)
            torch.autograd.backward(fm2, dm1, retain_graph=True)

        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            del dn2
            m2.grad = None

            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

        return x, m, dx, dm

class ReversibleCrossAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)        

    def forward(self, x, m, *, context, context_mask, video_mask = None, audio_mask = None, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        fn_context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with fn_context():
            y1 = x1 + self.f(x2, context = context, context_mask = context_mask, mask = video_mask, record_rng = record_rng)
            y2 = x2 + self.g(y1, record_rng = record_rng)
            n1 = m1 + self.j(m2, context = context, context_mask = context_mask, mask = audio_mask, record_rng = record_rng)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, *, context, context_mask, video_mask = None, audio_mask = None, **kwargs):
        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng = True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng = True, context = context, context_mask = context_mask, mask = video_mask)
            torch.autograd.backward(fx2, dx1, retain_graph = True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.k(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            fm2 = self.j(m2, set_rng = True, context = context, context_mask = context_mask, mask = audio_mask)
            torch.autograd.backward(fm2, dm1, retain_graph=True)

        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            del dn2
            m2.grad = None

            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

        return x, m, dx, dm

# reversible cross attention block

class ReversibleCrossModalityAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)

    def forward(self, x, m, *, video_mask = None, audio_mask = None, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        fn_context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with fn_context():
            y1 = x1 + self.f(x2, m2, record_rng = record_rng, mask = video_mask, context_mask = audio_mask)
            y2 = x2 + self.k(y1, record_rng = record_rng)
            n1 = m1 + self.j(m2, y2, record_rng = record_rng, mask = audio_mask, context_mask = video_mask)
            n2 = m2 + self.g(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, video_mask = None, audio_mask = None, **kwargs):
        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.g(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            y2.requires_grad = True
            fm2 = self.j(m2, y2, set_rng=True, mask = audio_mask, context_mask = video_mask)
            torch.autograd.backward(fm2, dm1)

        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            dx2 = dy2 + y2.grad
            del dn2
            del dy2
            m2.grad = None
            y2.grad = None

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.k(y1, set_rng = True)
            torch.autograd.backward(gy1, dx2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            m2.requires_grad = True
            fx2 = self.f(x2, m2, set_rng = True, mask = video_mask, context_mask = audio_mask)
            torch.autograd.backward(fx2, dx1)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dx2 + x2.grad
            dm2 = dm2 + m2.grad
            x2.grad = None
            m2.grad = None

        with torch.no_grad():
            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        return x, m, dx, dm

# reverse and non reverse functions

class ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, inp, ind, blocks, kwargs):
        x, m = split_at_index(1, ind, inp)

        for block in blocks:
            x, m = block(x, m, _reverse = True, **kwargs)

        ctx.blocks = blocks
        ctx.kwargs = kwargs
        ctx.ind = ind
        ctx.save_for_backward(x.detach(), m.detach())
        return torch.cat((x, m), dim = 1)

    @staticmethod
    def backward(ctx, d):
        ind = ctx.ind
        blocks = ctx.blocks
        kwargs = ctx.kwargs
        dy, dn = split_at_index(1, ind, d)
        y, n = ctx.saved_tensors

        for block in blocks[::-1]:
            y, n, dy, dn = block.backward_pass(y, n, dy, dn, **kwargs)

        d = torch.cat((dy, dn), dim = 1)
        return d, None, None, None

reversible_apply = ReversibleFunction.apply

def irreversible_apply(inputs, ind, blocks, kwargs):
    x, m = split_at_index(1, ind, inputs)
    for block in blocks:
        x, m = block(x, m, _reverse = False, **kwargs)
    return torch.cat((x, m), dim = 1)

# main reversible sequence class

class DualModalityReversibleSequence(nn.Module):
    def __init__(self, input_blocks, block_types):
        super().__init__()
        self.block_types = block_types
        blocks = nn.ModuleList([])

        for block, block_type in zip(input_blocks, block_types):
            if block_type == 'intra_modality_self_attn':
                reversible_klass = ReversibleSelfAttnBlock
            elif block_type == 'intra_modality_cross_attn':
                reversible_klass = ReversibleCrossAttnBlock
            elif block_type == 'inter_modality_cross_attn':
                reversible_klass = ReversibleCrossModalityAttnBlock
            else:                
                raise ValueError(f'unknown layer type {block_type}')

            blocks.append(reversible_klass(*block))

        self.blocks = blocks

    def forward(
        self,
        video,
        audio,
        *,
        context,
        context_mask = None,
        video_mask = None,
        audio_mask = None,
        reverse = True
    ):  
        blocks = self.blocks
        video, audio = list(map(lambda t: torch.cat((t, t), dim = -1), (video, audio)))
        kwargs = {'context': context, 'context_mask': context_mask, 'video_mask': video_mask, 'audio_mask': audio_mask}

        fn = reversible_apply if reverse else irreversible_apply
        ind = video.shape[1]
        inp = torch.cat((video, audio), dim = 1)
        out = fn(inp, ind, blocks, kwargs)
        video, audio  = split_at_index(1, ind, out)
        return list(map(lambda t: reduce(t, 'b n (c d) -> b n d', 'mean', c = 2), (video, audio)))
