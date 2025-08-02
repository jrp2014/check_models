import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

model_id = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="mps"
)
processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


picpath = "/Users/jrp/Pictures/Processed"
pics = sorted(Path(picpath).iterdir(), key=os.path.getmtime, reverse=True)
pic = str(pics[0])
print(pic)
image = Image.open(pic)

# you can resize the image here if it's not fitting to vram, or set model max sizes.
# image = image.resize((1024, 1024)) # like this

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
inputs = inputs.to("mps")

# with torch.no_grad():
#    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
output_ids = model.generate(
    **inputs, max_new_tokens=384, do_sample=True, temperature=0.7, use_cache=True, top_k=50
)


generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids, strict=False)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]
print(output_text)
