import contextlib
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import mlx.core as mx
from mlx import nn
from PIL.Image import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .models.base import BaseImageProcessor

MODEL_REMAPPING: dict[str, str]
MAX_FILE_SIZE_GB: int
generation_stream: Any

@contextlib.contextmanager
def wired_limit(
    model: nn.Module,
    streams: list[mx.Stream] | None = None,
) -> Generator[None, None, None]: ...

@dataclass
class GenerationResult:
    text: str
    token: int | None
    logprobs: list[float] | None
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory: float

def get_model_and_args(
    config: dict[str, Any],
) -> tuple[ModuleType, str | None]: ...

def get_model_path(
    path_or_hf_repo: str,
    revision: str | None = None,
) -> Path: ...

def load_model(
    model_path: Path,
    lazy: bool = False,
    **kwargs: dict[str, Any],
) -> nn.Module: ...

def sanitize_weights(
    model_obj: nn.Module,
    weights: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]: ...

def update_module_configs(
    model_config: dict[str, Any],
    model_class: type,
    config: dict[str, Any],
    modules: Sequence[str],
) -> dict[str, Any]: ...

def get_class_predicate(
    skip_vision: bool,
    weights: dict[str, Any] | None = None,
) -> Callable[[str], bool]: ...

def load(
    path_or_hf_repo: str,
    adapter_path: str | None = None,
    lazy: bool = False,
    **kwargs: dict[str, Any],
) -> tuple[nn.Module, PreTrainedTokenizer | PreTrainedTokenizerFast]: ...

def load_config(
    model_path: str | Path,
    **kwargs: dict[str, Any],
) -> dict[str, Any]: ...

def load_image_processor(
    model_path: str | Path,
    **kwargs: dict[str, Any],
) -> BaseImageProcessor: ...

def load_processor(
    model_path: str | Path,
    add_detokenizer: bool = False,
    eos_token_ids: list[int] | None = None,
    **kwargs: dict[str, Any],
) -> PreTrainedTokenizer | PreTrainedTokenizerFast: ...

def fetch_from_hub(
    model_path: Path,
    lazy: bool = False,
    **kwargs: dict[str, Any],
) -> tuple[nn.Module, dict[str, Any], PreTrainedTokenizer]: ...

def make_shards(
    weights: dict[str, Any],
    max_file_size_gb: int = ...,
) -> list[dict[str, Any]]: ...

def upload_to_hub(
    path: str,
    upload_repo: str,
    hf_path: str,
) -> None: ...

def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: list[int],
    penalty: float,
) -> mx.array: ...

def save_weights(
    save_path: str | Path,
    weights: dict[str, Any],
    *,
    donate_weights: bool = ...,
) -> None: ...

def quantize_model(
    model: nn.Module,
    config: dict[str, Any],
    q_group_size: int,
    q_bits: int,
    skip_vision: bool = ...,
) -> tuple[dict[str, Any], dict[str, Any]]: ...

def save_config(
    config: dict[str, Any],
    config_path: str | Path,
) -> None: ...

def dequantize_model(model: nn.Module) -> nn.Module: ...

def convert(
    hf_path: str,
    mlx_path: str | None = None,
    quantize: bool = False,
    q_group_size: int = ...,
    q_bits: int = ...,
    dtype: str = "float16",
    upload_repo: str | None = None,
    revision: str | None = None,
    dequantize: bool = False,
    skip_vision: bool = False,
    trust_remote_code: bool = False,
) -> None: ...

def load_image(
    image_source: str | Path | BytesIO,
    timeout: int = ...,
) -> Image: ...

def resize_image(
    img: Image,
    max_size: int,
) -> Image: ...

def process_image(
    img: Image,
    resize_shape: tuple[int, int],
    image_processor: BaseImageProcessor,
) -> Image: ...

def process_inputs(
    processor: PreTrainedTokenizer,
    images: list[Image],
    prompts: list[str],
    return_tensors: str = ...,
) -> dict[str, Any]: ...

def process_inputs_with_fallback(
    processor: PreTrainedTokenizer,
    images: list[Image],
    prompts: list[str],
    return_tensors: str = ...,
) -> dict[str, Any]: ...

def prepare_inputs(
    processor: PreTrainedTokenizer,
    images: list[Image],
    prompts: list[str],
    image_token_index: int,
    resize_shape: tuple[int, int] = ...,
) -> dict[str, Any]: ...

class TokenIdHandler:
    def __init__(
        self,
        eos_token_ids: list[int],
        tokenizer: PreTrainedTokenizer = ...,
    ) -> None: ...
    
    def add_eos_token_ids(
        self,
        new_eos_token_ids: int | list[int] = ...,
    ) -> None: ...
    
    def reset(
        self,
        eos_token_ids: list[int] = ...,
    ) -> None: ...

def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: str | list[str] = ...,
    **kwargs: dict[str, Any],
) -> str | Generator[str, None, None]: ...

def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: str | list[str] = ...,
    verbose: bool = ...,
    **kwargs: dict[str, Any],
) -> str: ...

