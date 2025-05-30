from typing import Any, TypedDict

class Message(TypedDict, total=False):
    role: str
    content: str | list[dict[str, Any]]

def get_message_json(
    model_name: str,
    prompt: str,
    role: str = ...,
    skip_image_token: bool = ...,
    num_images: int = ...,
    **kwargs: Any,
) -> Message:
    ...

def get_chat_template(
    processor: Any,
    messages: list[Message],
    add_generation_prompt: bool,
    tokenize: bool = ...,
    **kwargs: Any,
) -> str:
    ...

def apply_chat_template(
    processor: Any,
    config: dict[str, Any],
    prompt: str,
    add_generation_prompt: bool = ...,
    return_messages: bool = ...,
    num_images: int = ...,
    **kwargs: Any,
) -> str | list[Message]:
    ...

