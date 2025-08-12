from generate import load_model, prepare_inputs, generate_text

processor, model = load_model("llava-hf/llava-1.5-13b-hf")

max_tokens, temperature = 256, 0.1

prompt = "USER: <image>\nProvide a succinct caption, suitable for a microstock site, and some keywords for this photograph\nASSISTANT:"
# prompt = "USER: <image>\nProvide a decriptive caption and some keywords for this photograph\nASSISTANT:"
#prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nDescribe this picture<|im_end|><|im_start|>assistant\n"

image = "/Users/jrp/Pictures/Processed/20240420-154235_L1010558-HDR.jpg"
input_ids, pixel_values = prepare_inputs(processor, image, prompt)

reply = generate_text(
    input_ids, pixel_values, model, processor, max_tokens, temperature
    )
print(reply)
