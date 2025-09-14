import mlx.core as mx
from .conversation import get_conv_template as get_conv_template
from PIL import Image as Image
from _typeshed import Incomplete
from dataclasses import dataclass
from transformers import LlamaTokenizerFast as LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin
from typing import Literal

def select_best_resolution(image_size, candidate_resolutions): ...

class DictOutput:
    def keys(self): ...
    def __getitem__(self, item): ...
    def __setitem__(self, key, value) -> None: ...

@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: mx.array
    target_ids: mx.array
    images: mx.array
    images_seq_mask: mx.array
    images_spatial_crop: mx.array
    num_image_tokens: list[int]
    def __len__(self) -> int: ...

@dataclass
class BatchCollateOutput(DictOutput):
    sft_format: list[str]
    input_ids: mx.array
    labels: mx.array
    images: mx.array
    attention_mask: mx.array
    images_seq_mask: mx.array
    images_spatial_crop: mx.array
    seq_lens: list[int]

class ImageTransform:
    mean: Incomplete
    std: Incomplete
    normalize: Incomplete
    def __init__(self, mean: tuple[float, float, float] | None = (0.5, 0.5, 0.5), std: tuple[float, float, float] | None = (0.5, 0.5, 0.5), normalize: bool = True) -> None: ...
    def __call__(self, pil_img: Image.Image): ...

class DeepseekVLV2Processor(ProcessorMixin):
    tokenizer_class: Incomplete
    attributes: Incomplete
    candidate_resolutions: Incomplete
    image_size: Incomplete
    patch_size: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    normalize: Incomplete
    downsample_ratio: Incomplete
    image_transform: Incomplete
    tokenizer: Incomplete
    image_token_id: Incomplete
    image_token: Incomplete
    pad_token: Incomplete
    add_special_token: Incomplete
    sft_format: Incomplete
    mask_prompt: Incomplete
    ignore_id: Incomplete
    chat_template: Incomplete
    def __init__(self, tokenizer: LlamaTokenizerFast, candidate_resolutions: tuple[tuple[int, int]], patch_size: int, downsample_ratio: int, image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5), image_std: tuple[float, float, float] = (0.5, 0.5, 0.5), normalize: bool = True, image_token: str = '<image>', pad_token: str = '<｜▁pad▁｜>', add_special_token: bool = False, sft_format: str = 'deepseek', mask_prompt: bool = True, ignore_id: int = -100, **kwargs) -> None: ...
    @property
    def default_chat_template(self): ...
    @property
    def bos_id(self): ...
    @property
    def eos_id(self): ...
    @property
    def pad_id(self): ...
    def encode(self, text: str, bos: bool = True, eos: bool = False): ...
    def decode(self, t: list[int], **kwargs) -> str: ...
    def process_one(self, prompt: str = None, conversations: list[dict[str, str]] = None, images: list[Image.Image] = None, apply_sft_format: bool = False, inference_mode: bool = True, system_prompt: str = '', **kwargs): ...
    def pad_sequence(self, sequences, padding_value): ...
    def tokenize_with_images(self, conversation: str, images: list[Image.Image], bos: bool = True, eos: bool = True, cropping: bool = True): ...
    def batchify(self, sample_list: list[VLChatProcessorOutput], padding: Literal['left', 'right'] = 'left') -> BatchCollateOutput: ...
    def __call__(self, *, text: str = None, images: list[Image.Image] = None, apply_sft_format: bool = False, force_batchify: bool = True, inference_mode: bool = True, system_prompt: str = '', **kwargs): ...
