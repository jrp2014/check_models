import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig
from .config import ModelConfig as ModelConfig, PerceiverConfig as PerceiverConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path
from transformers import AutoConfig as AutoConfig

class Idefics2PerceiverAttention(nn.Module):
    n_heads: Incomplete
    n_kv_heads: Incomplete
    scale: Incomplete
    q_proj: Incomplete
    k_proj: Incomplete
    v_proj: Incomplete
    o_proj: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array, kv: mx.array, mask: mx.array | None = None, cache: tuple[mx.array, mx.array] | None = None) -> mx.array: ...

class Idefics2PerceiverLayer(nn.Module):
    hidden_size: Incomplete
    n_latents: Incomplete
    depth: Incomplete
    rms_norm_eps: Incomplete
    input_latents_norm: Incomplete
    input_context_norm: Incomplete
    self_attn: Incomplete
    post_attention_layernorm: Incomplete
    mlp: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array, hidden_states: mx.array, mask: mx.array | None = None) -> mx.array: ...

class Idefics2PerceiverResampler(nn.Module):
    hidden_size: Incomplete
    n_latents: Incomplete
    latents: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array, mask: mx.array | None = None): ...

class MLP(nn.Module):
    gate_proj: Incomplete
    down_proj: Incomplete
    up_proj: Incomplete
    def __init__(self, dim, hidden_dim, output_size) -> None: ...
    def __call__(self, x) -> mx.array: ...

class Idefics2Connector(nn.Module):
    modality_projection: Incomplete
    perceiver_resampler: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array, mask=None) -> mx.array: ...

class Model(nn.Module):
    model_type: Incomplete
    config: Incomplete
    vision_model: Incomplete
    language_model: Incomplete
    connector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, pixel_attention_mask: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None, **kwargs): ...
    def sanitize(self, weights): ...
