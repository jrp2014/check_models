import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path
from transformers import AutoConfig as AutoConfig

class MLP(nn.Module):
    proj: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x): ...

class Idefics3Connector(nn.Module):
    scale_factor: Incomplete
    modality_projection: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def pixel_shuffle(self, x, scale_factor: int = 2): ...
    def __call__(self, image_hidden_states): ...

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
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, cache=None, **kwargs): ...
    def sanitize(self, weights): ...
