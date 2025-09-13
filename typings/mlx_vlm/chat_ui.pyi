from .generate import stream_generate as stream_generate
from .prompt_utils import get_chat_template as get_chat_template, get_message_json as get_message_json
from .utils import load_config as load_config, load_image_processor as load_image_processor
from _typeshed import Incomplete
from collections.abc import Generator
from mlx_vlm import load as load

def parse_arguments(): ...

args: Incomplete
config: Incomplete
model: Incomplete
processor: Incomplete
image_processor: Incomplete

def chat(message, history, temperature, max_tokens) -> Generator[Incomplete]: ...

demo: Incomplete
