import mlx.core as mx
import mlx.nn as nn
from ..base import BaseModelConfig as BaseModelConfig, expand2square as expand2square
from .config import ModelConfig as ModelConfig, ProjectorConfig as ProjectorConfig
from .language import LanguageModel as LanguageModel, TextConfig as TextConfig
from .processing_deepsek_vl_v2 import DeepseekVLV2Processor as DeepseekVLV2Processor
from .vision import VisionConfig as VisionConfig, VisionModel as VisionModel
from PIL import Image as Image
from _typeshed import Incomplete
from dataclasses import dataclass as dataclass
from huggingface_hub import snapshot_download as snapshot_download
from pathlib import Path as Path
from transformers.image_processing_utils import BaseImageProcessor as BaseImageProcessor, BatchFeature as BatchFeature
from transformers.image_utils import to_numpy_array as to_numpy_array

class MlpProjector(nn.Module):
    config: Incomplete
    token_pooling_layer: Incomplete
    layers: Incomplete
    def __init__(self, config: ProjectorConfig) -> None: ...
    def __call__(self, x): ...

class Model(nn.Module):
    config: Incomplete
    vision: Incomplete
    language_model: Incomplete
    projector: Incomplete
    vision_feature_layer: Incomplete
    vision_feature_select_strategy: Incomplete
    tile_tag: Incomplete
    global_view_pos: Incomplete
    image_newline: Incomplete
    view_separator: Incomplete
    tile_indicators: Incomplete
    def __init__(self, config: ModelConfig) -> None: ...
    def process_image_features(self, input_embeds, images_embeds, images_spatial_crop, images_seq_mask, h, w, n_dim): ...
    def get_input_embeddings(self, input_ids: mx.array | None = None, pixel_values: mx.array | None = None, images_spatial_crop: mx.array | None = None, image_seq_mask: mx.array | None = None): ...
    @property
    def layers(self): ...
    def __call__(self, input_ids: mx.array, pixel_values: mx.array | None = None, mask: mx.array | None = None, cache=None, **kwargs): ...
    @staticmethod
    def sanitize(weights): ...
