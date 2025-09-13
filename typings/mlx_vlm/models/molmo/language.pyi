import mlx.core as mx
import mlx.nn as nn
from ..base import LanguageModelOutput as LanguageModelOutput, create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from ..cache import KVCache as KVCache
from .config import TextConfig as TextConfig
from _typeshed import Incomplete

class SwiGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array: ...

class MolmoBlock(nn.Module):
    attn_out: Incomplete
    ff_out: Incomplete
    attn_norm: Incomplete
    ff_norm: Incomplete
    ff_proj: Incomplete
    rotary_emb: Incomplete
    scale: Incomplete
    n_heads: Incomplete
    n_kv_heads: Incomplete
    fused_dims: Incomplete
    att_proj: Incomplete
    act: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, x, mask=None, cache=None): ...

class Embedding(nn.Module):
    initializer_range: Incomplete
    new_embed_initializer_range: Incomplete
    embedding: Incomplete
    new_embedding: Incomplete
    def __init__(self, num_embeddings: int, num_new_embeddings: int, features: int, initializer_range: float = 0.02, new_embed_initializer_range: float = 0.02) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Molmo(nn.Module):
    config: Incomplete
    wte: Incomplete
    blocks: Incomplete
    ln_f: Incomplete
    ff_out: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache: KVCache | None = None) -> LanguageModelOutput: ...

class LanguageModel(nn.Module):
    config: Incomplete
    model_type: Incomplete
    model: Incomplete
    def __init__(self, config: TextConfig) -> None: ...
    def __call__(self, input_ids: mx.array, inputs_embeds: mx.array | None = None, mask: mx.array | None = None, cache: KVCache | None = None) -> LanguageModelOutput: ...
    @staticmethod
    def sanitize(weights): ...
    @property
    def layers(self): ...
    @property
    def head_dim(self): ...
    @property
    def n_kv_heads(self): ...
