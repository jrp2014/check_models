import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class Model(nn.Module):
    config: Incomplete
    language_model: Incomplete
    vision_tower: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None, **kwargs) -> dict[str, mx.array | list[tuple[mx.array, mx.array]]]: ...
    def sanitize(self, weights): ...
