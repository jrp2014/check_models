import mlx.core as mx
import mlx.nn as nn
from .config import VisionConfig as VisionConfig
from _typeshed import Incomplete

class MLP(nn.Module):
    config: Incomplete
    hidden_size: Incomplete
    w1: Incomplete
    w2: Incomplete
    w3: Incomplete
    def __init__(self, config: VisionConfig, input_dim: int) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class ViTMLP(nn.Module):
    config: Incomplete
    w1: Incomplete
    w2: Incomplete
    act: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class MultiHeadDotProductAttention(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    num_key_value_heads: Incomplete
    num_key_value_groups: Incomplete
    scale: Incomplete
    is_vit_layer: Incomplete
    wq: Incomplete
    wk: Incomplete
    wv: Incomplete
    wo: Incomplete
    def __init__(self, config: VisionConfig, is_vit_layer: bool | None = True) -> None: ...
    def __call__(self, x: mx.array, kv: mx.array = None) -> mx.array: ...

class ResidualAttentionBlock(nn.Module):
    config: Incomplete
    attention: Incomplete
    feed_forward: Incomplete
    attention_norm: Incomplete
    ffn_norm: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class ResidualAttentionBlocks(nn.Module):
    resblocks: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def pad_to_multiple(x, target_size, pad_mode: str = 'edge', pad_value: int = 0): ...

class VisionTransformer(nn.Module):
    config: Incomplete
    class_embedding: Incomplete
    positional_embedding: Incomplete
    patch_embedding: Incomplete
    pre_ln: Incomplete
    transformer: Incomplete
    def __init__(self, config: VisionConfig) -> None: ...
    def add_pos_emb(self, x: mx.array, patch_num: int) -> mx.array: ...
    def __call__(self, x: mx.array, patch_num: int = None) -> list[mx.array]: ...

class VisionModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    image_vit: Incomplete
    num_prefix_tokens: int
    image_pooling_2d: Incomplete
    image_projector: Incomplete
    pad_embed: Incomplete
    def __init__(self, config) -> None: ...
    def encode_image(self, images: mx.array) -> mx.array: ...
    def __call__(self, images: mx.array, image_masks: mx.array) -> tuple[mx.array, mx.array | None]: ...
