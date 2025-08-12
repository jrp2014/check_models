model_path = "mlx-community/deepseek-vl2-8bit"

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import subprocess
import time
import psutil


print("\033[1mRunning", model_path, "\033[0m")

process = psutil.Process()
mem_before = process.memory_info().rss

try:
    # Load the model
    model, tokenizer = load(model_path)
    config = load_config(model_path)
except Exception as e:
    print(f"Failed to load model at {model_path}: {e}")

# Prepare input
image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(
    tokenizer, config, prompt, num_images=len(image)
)

# Generate output
try:
    start_time = time.time()
    output = generate(model, tokenizer, image, formatted_prompt, verbose=True)
    end_time = time.time()
    print(output)
except Exception as e:
    print(f"Failed to generate output for model at {model_path}: {e}")

mem_after = process.memory_info().rss
print(f"Output generated in {end_time - start_time:.2f}s")
print(f"Memory used: {(mem_after - mem_before) / (1024 * 1024 * 1024):.2f} GB")

print(80 * "-", end="\n\n")
