import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig
from ..cache import KVCache as KVCache
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class Model(nn.Module):
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    multi_modal_projector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache: KVCache | None = None, **kwargs) -> tuple[mx.array, mx.array | None]: ...
    def sanitize(self, weights): ...
