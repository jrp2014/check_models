import abc
import mlx.core as mx
from _typeshed import Incomplete
from abc import abstractmethod
from dataclasses import dataclass
from mlx_lm.models.base import create_attention_mask as create_attention_mask, scaled_dot_product_attention as scaled_dot_product_attention
from mlx_lm.models.cache import RotatingKVCache as RotatingKVCache
from transformers.image_processing_utils import BaseImageProcessor as ImageProcessor

@dataclass
class LanguageModelOutput:
    logits: mx.array
    cross_attention_states: list[mx.array] | None = ...
    encoder_outputs: list[mx.array] | None = ...

@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params): ...

class BaseImageProcessor(ImageProcessor, metaclass=abc.ABCMeta):
    image_mean: Incomplete
    image_std: Incomplete
    size: Incomplete
    resample: Incomplete
    rescale_factor: Incomplete
    data_format: Incomplete
    crop_size: Incomplete
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: dict[str, int] = None, resample=..., rescale_factor=..., data_format=...) -> None: ...
    @abstractmethod
    def preprocess(self, images): ...

def expand2square(pil_img, background_color): ...
def check_array_shape(arr): ...
def check_activation_stats(name, tensor) -> None: ...
def pixel_shuffle(input_tensor, shuffle_ratio): ...
def interpolate(pos_embed, size, mode: str = 'cubic', align_corners: bool = False): ...
