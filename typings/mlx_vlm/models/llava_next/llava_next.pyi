import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class LlavaMultiModalProjector(nn.Module):
    linear_1: Incomplete
    gelu: Incomplete
    linear_2: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Model(nn.Module):
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    image_newline: Incomplete
    multi_modal_projector: Incomplete
    vision_feature_layer: Incomplete
    vision_feature_select_strategy: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None, **kwargs): ...
