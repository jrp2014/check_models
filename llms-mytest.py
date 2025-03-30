from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3.1-70B-bf16")
response = generate(model, tokenizer, prompt="hello", verbose=True)
