import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask
from ..cache import KVCache as KVCache
from .config import ModelConfig as ModelConfig, TextConfig as TextConfig
from .language import LanguageModel as LanguageModel
from .su_rope import Phi3SuScaledRotaryEmbedding as Phi3SuScaledRotaryEmbedding
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from types import SimpleNamespace as SimpleNamespace

class Attention(nn.Module):
    n_heads: Incomplete
    n_kv_heads: Incomplete
    num_hidden_layers: Incomplete
    head_dim: Incomplete
    scale: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rope: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class MLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, dim, hidden_dim) -> None: ...
    def __call__(self, x) -> mx.array: ...

class TransformerBlock(nn.Module):
    num_attention_heads: Incomplete
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    config: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array: ...

class Phi3V(nn.Module):
    config: Incomplete
    vocab_size: Incomplete
    num_hidden_layers: Incomplete
    embed_tokens: Incomplete
    vision_embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, pixel_values=None, image_sizes=None, mask: mx.array | None = None, cache=None): ...

class Model(nn.Module):
    model_type: Incomplete
    model: Incomplete
    lm_head: Incomplete
    config: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, inputs: mx.array, pixel_values=None, mask=None, cache=None, image_sizes=None, **kwargs): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
    @property
    def language_model(self): ...
    @property
    def vision_model(self): ...
