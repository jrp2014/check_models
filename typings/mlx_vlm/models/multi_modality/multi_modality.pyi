import mlx.core as mx
import mlx.nn as nn
import numpy as np
from ..base import BaseImageProcessor as BaseImageProcessor, expand2square as expand2square
from .config import ModelConfig as ModelConfig, ProjectorConfig as ProjectorConfig, TextConfig as TextConfig, VisionConfig as VisionConfig
from .language import LanguageModel as LanguageModel
from .vision import VisionModel as VisionModel
from PIL import Image
from _typeshed import Incomplete
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path
from transformers.image_processing_utils import BatchFeature as BatchFeature

class ImageProcessor(BaseImageProcessor):
    model_input_names: Incomplete
    image_size: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    do_normalize: bool
    rescale_factor: Incomplete
    min_size: Incomplete
    background_color: Incomplete
    def __init__(self, config, image_size: int = 384, min_size: int = 14, image_mean: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.5), image_std: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.5), rescale_factor: float = ..., do_normalize: bool = True, **kwargs) -> None: ...
    def resize(self, pil_img: Image) -> np.ndarray: ...
    def preprocess(self, images, **kwargs) -> BatchFeature: ...

class MlpProjector(nn.Module):
    layers: Incomplete
    high_up_proj: Incomplete
    low_up_proj: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def __call__(self, x: mx.array | tuple) -> mx.array: ...

class Model(nn.Module):
    config: Incomplete
    vision_model: Incomplete
    language_model: Incomplete
    aligner: Incomplete
    vision_feature_layer: Incomplete
    vision_feature_select_strategy: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def add_image_token(self, image_indices: list, input_ids: np.ndarray, image_token_index: int, num_image_tokens: int, add_special_token: bool = False): ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array, mask: mx.array, cache=None, **kwargs): ...
