import mlx.core as mx
import mlx.nn as nn
from .config import ModelConfig as ModelConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path

class PaliGemmaMultiModalProjector(nn.Module):
    linear: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

class Model(nn.Module):
    model_type: Incomplete
    config: Incomplete
    vision_tower: Incomplete
    language_model: Incomplete
    multi_modal_projector: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, mask: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array | None = None, cache: mx.array | None = None, **kwargs): ...
