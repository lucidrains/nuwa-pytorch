import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# main class

class NUWA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
