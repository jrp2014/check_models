from _typeshed import Incomplete
from enum import Enum
from typing import Any

class MessageFormat(Enum):
    LIST_WITH_IMAGE = 'list_with_image'
    LIST_WITH_IMAGE_FIRST = 'list_with_image_first'
    LIST_WITH_IMAGE_TYPE = 'list_with_image_type'
    LIST_WITH_IMAGE_TYPE_TEXT = 'list_with_image_type_text'
    LIST_WITH_IMAGE_TYPE_TEXT_IMAGE_LAST = 'list_with_image_type_text_image_last'
    IMAGE_TOKEN = 'image_token'
    IMAGE_TOKEN_PIPE = 'image_token_pipe'
    START_IMAGE_TOKEN = 'start_image_token'
    IMAGE_TOKEN_NEWLINE = 'image_token_newline'
    NUMBERED_IMAGE_TOKENS = 'numbered_image_tokens'
    PROMPT_ONLY = 'prompt_only'
    PROMPT_WITH_IMAGE_TOKEN = 'prompt_with_image_token'
    PROMPT_WITH_START_IMAGE_TOKEN = 'prompt_with_start_image_token'
    VIDEO_WITH_TEXT = 'video_with_text'

MODEL_CONFIG: Incomplete
SINGLE_IMAGE_ONLY_MODELS: Incomplete

class MessageBuilder:
    @staticmethod
    def text_message(text: str) -> dict[str, str]: ...
    @staticmethod
    def content_message(content: str) -> dict[str, str]: ...
    @staticmethod
    def image_message() -> dict[str, str]: ...
    @staticmethod
    def audio_message() -> dict[str, str]: ...
    @staticmethod
    def video_message(video_path: str, max_pixels: int = ..., fps: int = 1) -> dict[str, Any]: ...

class MessageFormatter:
    model_name: Incomplete
    format_type: Incomplete
    def __init__(self, model_name: str) -> None: ...
    def format_message(self, prompt: str, role: str = 'user', skip_image_token: bool = False, skip_audio_token: bool = False, num_images: int = 1, num_audios: int = 1, **kwargs) -> str | dict[str, Any]: ...

def get_message_json(model_name: str, prompt: str, role: str = 'user', skip_image_token: bool = False, skip_audio_token: bool = False, num_images: int = 0, num_audios: int = 0, **kwargs) -> str | dict[str, Any]: ...
def get_chat_template(processor, messages: list[dict[str, Any]], add_generation_prompt: bool, tokenize: bool = False, **kwargs) -> Any: ...
def apply_chat_template(processor, config: dict[str, Any] | Any, prompt: str | dict[str, Any] | list[Any], add_generation_prompt: bool = True, return_messages: bool = False, num_images: int = 0, num_audios: int = 0, **kwargs) -> list[dict[str, Any]] | str | Any: ...
