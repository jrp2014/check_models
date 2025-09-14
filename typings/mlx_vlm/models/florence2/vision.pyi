import mlx.core as mx
import mlx.nn as nn
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete
from collections import OrderedDict as OrderedDict

def check_array_shape(arr): ...

class MlpFC(nn.Module):
    fc1: Incomplete
    fc2: Incomplete
    gelu: Incomplete
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None) -> None: ...
    def __call__(self, x): ...

class Mlp(nn.Module):
    net: Incomplete
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None) -> None: ...
    def __call__(self, x, size): ...

class DepthWiseConv2d(nn.Module):
    dw: Incomplete
    def __init__(self, dim_in: int, kernel_size: int, padding: int, stride: int, bias: bool = True) -> None: ...
    def __call__(self, x, size): ...

class ConvEmbed(nn.Module):
    proj: Incomplete
    norm: Incomplete
    pre_norm: Incomplete
    def __init__(self, patch_size: int = 7, in_chans: int = 3, embed_dim: int = 64, stride: int = 4, padding: int = 2, norm_layer: nn.Module | None = None, pre_norm: bool = True) -> None: ...
    def __call__(self, x, size): ...

class ChannelAttention(nn.Module):
    groups: Incomplete
    qkv: Incomplete
    proj: Incomplete
    def __init__(self, dim: int, groups: int = 8, qkv_bias: bool = True) -> None: ...
    def __call__(self, x, size): ...

def window_partition(x: mx.array, window_size: int): ...
def window_reverse(windows: mx.array, batch_size: int, window_size: int, H: int, W: int): ...

class WindowAttention(nn.Module):
    dim: Incomplete
    window_size: Incomplete
    num_heads: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    def __init__(self, dim: int, num_heads: int, window_size: int, qkv_bias: bool = True) -> None: ...
    def __call__(self, x, size): ...

class PreNorm(nn.Module):
    norm: Incomplete
    fn: Incomplete
    drop_path: Incomplete
    def __init__(self, norm, fn, drop_path=None) -> None: ...
    def __call__(self, x, size): ...

class DropPath(nn.Module):
    drop_prob: Incomplete
    def __init__(self, drop_prob: float = 0.0) -> None: ...
    def __call__(self, x): ...

class SpatialBlock(nn.Module):
    conv1: Incomplete
    window_attn: Incomplete
    conv2: Incomplete
    ffn: Incomplete
    def __init__(self, dim: int, num_heads: int, window_size: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_path_rate: float = 0.0, conv_at_attn: bool = True, conv_at_ffn: bool = True) -> None: ...
    def __call__(self, x, size): ...

class ChannelBlock(nn.Module):
    conv1: Incomplete
    channel_attn: Incomplete
    conv2: Incomplete
    ffn: Incomplete
    def __init__(self, dim: int, groups: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_path_rate: float = 0.0, conv_at_attn: bool = True, conv_at_ffn: bool = True) -> None: ...
    def __call__(self, x, size): ...

class Block(nn.Module):
    spatial_block: Incomplete
    channel_block: Incomplete
    def __init__(self, dim: int, num_heads: int, num_groups: int, window_size: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_path_rate: tuple[float, float] = (0.0, 0.0), conv_at_attn: bool = True, conv_at_ffn: bool = True) -> None: ...
    def __call__(self, x, size): ...

class VisionModel(nn.Module):
    num_classes: Incomplete
    model_type: Incomplete
    dim_embed: Incomplete
    num_heads: Incomplete
    num_groups: Incomplete
    num_stages: Incomplete
    convs: Incomplete
    blocks: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x): ...
    @staticmethod
    def sanitize(weights): ...
