import os
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config  # unused?

# https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3   # Very concise, no keywords
# model_path = "HuggingFaceTB/SmolVLM-Instruct" # Fast, but too concise (eg, no keywords)
# model_path = "OpenGVLab/InternVL2_5-38B" # ValueError: Model type internvl_chat not supported.
# model_path = "Qwen/Qwen2-VL-7B-Instruct"  # libc++abi: terminating due to uncaught exception of type std::runtime_error: Attempting to allocate 269535412224 bytes which is greater than the maximum allowed buffer size of 28991029248 bytes.###
# model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"  # needs about 95Gb, precise, but is slow
# model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct" # Very slow, gives more detailed captions, but is not always accurate.  Uses over 90Gb.  No keywords.
# model_path = "mistral-community/pixtral-12b"
# model_path = "mlx-community/Phi-3.5-vision-instruct-bf16" # OK, but doesn't provide keywords
# model_path = "mlx-community/llava-1.5-7b-4bit"
model_path = "mlx-community/llava-v1.6-34b-8bit"  # Slower but more precise
# model_path = "mlx-community/llava-v1.6-mistral-7b-8bit"
# model_path = "mlx-community/pixtral-12b-8bit" # To the point
# model_path ="mlx-community/Qwen2-VL-72B-Instruct-8bit" # libc++abi: terminating due to uncaught exception of type std::runtime_error: Attempting to allocate 135383101952 bytes which is greater than the maximum allowed buffer size of 77309411328 bytes.
# model_path ="mlx-community/dolphin-vision-72b-4bit"  # Needs image_processor = load_image_processor(model_path)
# model_path="JosefAlbers/akemiH_MedQA_Reason"
# model_path="Qwen/Qwen2-VL-7B-Instruct" # breaks with new mlx
# model_path="cognitivecomputations/dolphin-2.9.2-qwen2-72b" #  No module named 'mlx_vlm.models.qwen2'
# model_path="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# model_path="google/siglip-so400m-patch14-384" # Model type siglip not supported.
# model_path="meta-llama/Llama-3.2-11B-Vision-Instruct" # Unusably slow, gives fairly detailed captions, but is not always accurate.  Uses over 90Gb.  No keywords.
# model_path="meta-llama/Llama-3.2-90B-Vision-Instruct"
# model_path="microsoft/Phi-3.5-mini-instruct"
# model_path="microsoft/Phi-3.5-vision-instruct" # provides a good description, but that is all
# model_path="mistral-community/pixtral-12b" # Unsupported model type: pixtral
# model_path="mlx-community/Florence-2-large-ft-bf16" # Produces gibberish v quickly.  Corrupt?
# model_path="mlx-community/Llama-3.2-11B-Vision-Instruct-8bit" # Much better than the native version, but still slows down
# model_path="mlx-community/Molmo-7B-D-0924-bf16"  ModuleNotFoundError: No module named 'einops'
# model_path="mlx-community/Phi-3.5-vision-instruct-bf16" # Pretty good description, but no keywords
# model_path="mlx-community/Qwen2-VL-72B-Instruct-8bit" # libc++abi: terminating due to uncaught exception of type std::runtime_error: Attempting to allocate 135383101952 bytes which is greater than the maximum allowed buffer size of 77309411328 bytes.
# model_path="mlx-community/SmolVLM-Instruct-bf16" # Very good, fast description, but that is all
# model_path="mlx-community/dolphin-vision-72b-4bit" # mlx-vlm crash
# model_path="mlx-community/idefics2-8b-chatty-8bit" # ValueError: Unsupported model type: idefics2_vision
# model_path="mlx-community/llava-1.5-7b-4bit" # Vague description
# model_path="mlx-community/llava-v1.6-34b-8bit" # Pretty good
# model_path="mlx-community/llava-v1.6-mistral-7b-8bit" # V similar to the above
# model_path="mlx-community/paligemma2-10b-ft-docci-448-bf16" # Generates a good description, but no keywords
# model_path="mlx-community/paligemma2-3b-pt-896-4bit" # too innaccurate
# model_path="mlx-community/pixtral-12b-8bit" # A rival to llava 1.6
# model_path="mlx-community/whisper-tiny"
# model_path = "mlx-community/paligemma2-10b-ft-docci-448-bf16" # Detailed description, but no keywords
# model_path = "mlx-community/QVQ-72B-Preview-8bit" # libc++abi: terminating due to uncaught exception of type std::runtime_error: Attempting to allocate 134767706112 bytes which is greater than the maximum allowed buffer size of 77309411328 bytes.
# model_path = "mlx-community/deepseek-vl2-8bit" # Proves a fair description, but no keywords

print("Model: ", model_path)

# Load the model
model, processor = load(model_path)
# processor = load_image_processor(model_path)
config = load_config(model_path)

prompt = "Provide a factual caption, description and comma-separated keywords or tags for this image so that it can be searched for easily"

picpath = "/Users/jrp/Pictures/Processed"
pics = sorted(Path(picpath).iterdir(), key=os.path.getmtime, reverse=True)
pic = str(pics[0])
print("Image: ", pic)

# Apply chat template
formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

# Generate output
output = generate(model, processor, formatted_prompt, pic, max_tokens=500, verbose=True)
print(output)
