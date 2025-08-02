from generate import generate_text, load_model, prepare_inputs

# model_path = "mlx-community/llava-1.5-7b-4bit"
model_path = "mlx-community/llava-v1.6-mistral-7b-8bit"
# model_path = "mlx-community/llava-v1.6-34b-8bit"

model, processor = load_model(model_path)

max_tokens, temperature = 128, 0.0

prompt = "USER: <image>\nProvide a caption and keywords for this image\nASSISTANT:"
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_ids, pixel_values = prepare_inputs(processor, image, prompt)

reply = generate_text(input_ids, pixel_values, model, processor, max_tokens, temperature)

print(reply)
