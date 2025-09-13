import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig
from .config import MLPConfig as MLPConfig, VisionConfig as VisionConfig
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass

def check_array_shape(arr): ...

class AttentionPoolLatent(nn.Module):
    num_heads: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    pool: Incomplete
    latent_dim: Incomplete
    latent_len: Incomplete
    latent: Incomplete
    q: Incomplete
    kv: Incomplete
    q_norm: Incomplete
    k_norm: Incomplete
    proj: Incomplete
    proj_drop: Incomplete
    pos_embed: Incomplete
    norm: Incomplete
    mlp: Incomplete
    def __init__(self, in_features: int, out_features: int = None, embed_dim: int = None, num_heads: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_norm: bool = False, latent_len: int = 1, latent_dim: int = None, pos_embed: str = '', pool_type: str = 'token', norm_layer: nn.Module | None = None, drop: float = 0.0) -> None: ...
    def __call__(self, x: mx.array): ...

class Attention(nn.Module):
    num_heads: Incomplete
    scale: Incomplete
    qkv: Incomplete
    proj: Incomplete
    def __init__(self, dims: int, num_heads: int, qkv_bias: bool = True) -> None: ...
    def __call__(self, x, mask=None): ...

class MLP(nn.Module):
    activation_fn: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    def __init__(self, config: VisionConfig | dict, bias: bool = True) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class EncoderLayer(nn.Module):
    embed_dim: Incomplete
    attn: Incomplete
    norm1: Incomplete
    mlp: Incomplete
    norm2: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array: ...

class VisionEmbeddings(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    proj: Incomplete
    num_patches: Incomplete
    num_positions: Incomplete
    norm: Incomplete
    def __init__(self, config: VisionConfig, norm_layer: bool = False) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class SigLipVisionModel(nn.Module):
    num_prefix_tokens: int
    no_embed_class: bool
    dynamic_img_size: bool
    ignore_head: Incomplete
    cls_token: Incomplete
    reg_token: Incomplete
    patch_embed: Incomplete
    norm_pre: Incomplete
    blocks: Incomplete
    norm: Incomplete
    pos_embed: Incomplete
    attn_pool: Incomplete
    def __init__(self, config: VisionConfig, ignore_head: bool, pre_norm: bool = False, no_embed_class: bool = True) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...

class VisionModel(nn.Module):
    model_type: Incomplete
    config: Incomplete
    vision_tower: Incomplete
    def __init__(self, config: VisionConfig, ignore_head: bool = True) -> None: ...
    def __call__(self, x: mx.array, output_hidden_states: bool | None = None) -> mx.array: ...
    def sanitize(self, weights): ...
