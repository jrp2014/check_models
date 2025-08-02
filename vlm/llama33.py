from mlx_lm import generate, load

model, tokenizer = load("mlx-community/Llama-3.3-70B-Instruct-8bit")

prompt = "Can you describei, caption, and keyword pictures?"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

response = generate(model, tokenizer, prompt=prompt, verbose=True)
