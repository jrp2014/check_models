import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="mps",
)
processor = AutoProcessor.from_pretrained(model_id)

picpath = "/Users/jrp/Pictures/Processed"
pics = sorted(Path(picpath).iterdir(), key=os.path.getmtime, reverse=True)
pic = str(pics[0])
print(pic)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open(pic)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            #       {"type": "text", "text": "Provide a caption and a list of keywords for this image, suitable for microstock."}
            #       {"type": "text", "text": "Analyze this image and provide a title (max 70 characters), description (max 200 characters), and comma-separated keywords suitable for microstock photography websites."}
            {
                "type": "text",
                "text": "You are an AI assistant that helps people craft a clear and detailed sentence that describes the content depicted in an image.  Then generate a list of descriptive, comma-separated tags for the following image. Analyze the image carefully and produce tags that accurately represent the image. Ensure the tags are relevant.",
            },
        ],
    }
]
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(output[0]))
