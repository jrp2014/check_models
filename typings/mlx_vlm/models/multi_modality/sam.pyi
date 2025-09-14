import mlx.core as mx
import mlx.nn as nn
import numpy as np
from .config import SAMViTCfg as SAMViTCfg
from _typeshed import Incomplete

class MLPBlock(nn.Module):
    lin1: Incomplete
    lin2: Incomplete
    act: Incomplete
    def __init__(self, embedding_dim: int, mlp_dim: int, act: type[nn.Module] = ...) -> None: ...
    def __call__(self, x: mx.array): ...

def resize_image(image_np, new_size=(96, 96), order: int = 1): ...

class SAMEncoder(nn.Module):
    img_size: Incomplete
    patch_embed: Incomplete
    pos_embed: Incomplete
    blocks: Incomplete
    neck: Incomplete
    downsamples: Incomplete
    sam_hd: bool
    hd_alpha_downsamples: Incomplete
    neck_hd: Incomplete
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, out_chans: int = 256, qkv_bias: bool = True, norm_layer: type[nn.Module] = ..., act_layer: type[nn.Module] = ..., use_abs_pos: bool = True, use_rel_pos: bool = True, rel_pos_zero_init: bool = True, window_size: int = 14, global_attn_indexes: tuple[int, ...] = (2, 5, 8, 11), downsample_channels: tuple[int, ...] = (512, 1024)) -> None: ...
    def __call__(self, x: mx.array): ...

class Block(nn.Module):
    norm1: Incomplete
    attn: Incomplete
    norm2: Incomplete
    mlp: Incomplete
    window_size: Incomplete
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, norm_layer: type[nn.Module] = ..., act_layer: type[nn.Module] = ..., use_rel_pos: bool = False, rel_pos_zero_init: bool = True, window_size: int = 0, input_size: tuple[int, int] | None = None) -> None: ...
    def __call__(self, x: mx.array): ...

class Attention(nn.Module):
    num_heads: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    use_rel_pos: Incomplete
    rel_pos_h: Incomplete
    rel_pos_w: Incomplete
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, use_rel_pos: bool = False, rel_pos_zero_init: bool = True, input_size: tuple[int, int] | None = None) -> None: ...
    def __call__(self, x: mx.array): ...

def window_partition(x: np.ndarray, window_size: int) -> tuple[np.ndarray, tuple[int, int]]: ...
def window_unpartition(windows: np.ndarray, window_size: int, pad_hw: tuple[int, int], hw: tuple[int, int]): ...
def get_rel_pos(q_size: int, k_size: int, rel_pos: np.ndarray) -> np.ndarray: ...
def add_decomposed_rel_pos(attn: np.ndarray, q: np.ndarray, rel_pos_h: np.ndarray, rel_pos_w: np.ndarray, q_size: tuple[int, int], k_size: tuple[int, int]) -> np.ndarray: ...

class PatchEmbed(nn.Module):
    proj: Incomplete
    def __init__(self, kernel_size: tuple[int, int] = (16, 16), stride: tuple[int, int] = (16, 16), in_chans: int = 3, embed_dim: int = 768) -> None: ...
    def __call__(self, x: mx.array): ...
