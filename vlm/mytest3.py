from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
model_path = "mlx-community/pixtral-12b-8bit"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
image: list[str] = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
prompt: str = "Describe this image."

# Apply chat template
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=len(image))

# Generate output
output = generate(model, processor, formatted_prompt, image, verbose=True, max_tokens=500)
print(output)
