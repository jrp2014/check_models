from .prompt_utils import apply_chat_template as apply_chat_template
from .trainer import Dataset as Dataset, Trainer as Trainer, save_adapter as save_adapter
from .trainer.utils import apply_lora_layers as apply_lora_layers, find_all_linear_names as find_all_linear_names, get_peft_model as get_peft_model
from .utils import load as load, load_image_processor as load_image_processor
from _typeshed import Incomplete

logger: Incomplete

def custom_print(*args, **kwargs) -> None: ...
def main(args): ...
