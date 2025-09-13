import mlx.core as mx
from PIL import Image
from _typeshed import Incomplete
from pathlib import Path as Path
from transformers import PreTrainedTokenizerBase as PreTrainedTokenizerBase, ProcessorMixin
from transformers.image_utils import ImageFeatureExtractionMixin

logger: Incomplete
IMAGENET_MEAN: Incomplete
IMAGENET_STD: Incomplete
chat_template: str
IMG_START_TOKEN: str
IMG_END_TOKEN: str
IMG_CONTEXT_TOKEN: str

def build_transform(input_size): ...
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size): ...
def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False): ...

class InternVLImageProcessor(ImageFeatureExtractionMixin):
    model_input_names: Incomplete
    do_resize: Incomplete
    size: Incomplete
    resample: Incomplete
    do_center_crop: Incomplete
    crop_size: Incomplete
    do_rescale: Incomplete
    rescale_factor: Incomplete
    do_normalize: Incomplete
    image_mean: Incomplete
    image_std: Incomplete
    do_dynamic_preprocess: Incomplete
    dynamic_min_num: Incomplete
    dynamic_max_num: Incomplete
    dynamic_use_thumbnail: Incomplete
    def __init__(self, do_resize: bool = True, size: int = 448, resample=..., do_center_crop: bool = False, crop_size=None, do_rescale: bool = True, rescale_factor: float = ..., do_normalize: bool = True, image_mean=..., image_std=..., do_dynamic_preprocess: bool = True, dynamic_min_num: int = 1, dynamic_max_num: int = 12, dynamic_use_thumbnail: bool = True, **kwargs) -> None: ...
    def preprocess(self, images: list[Image.Image], do_dynamic_preprocess: bool | None = None, size: int | None = None, return_tensors: str | None = None, **kwargs) -> list[mx.array]: ...

class InternVLChatProcessor(ProcessorMixin):
    attributes: Incomplete
    image_processor_class: str
    tokenizer_class: Incomplete
    num_image_token: Incomplete
    def __init__(self, image_processor=None, tokenizer=None, chat_template=..., **kwargs) -> None: ...
    def __call__(self, text: str | list[str] = None, images: list[Image.Image] = None, padding: bool | str = True, truncation: bool = True, max_length: int | None = None, return_tensors: str | None = 'pt', **kwargs): ...
    def batch_decode(self, *args, **kwargs): ...
    def decode(self, *args, **kwargs): ...
    def save_pretrained(self, save_directory, **kwargs) -> None: ...
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, **kwargs): ...

MODEL_TYPE: str
