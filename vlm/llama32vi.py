from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

messages = [
	{ "role": "user", "content": {"type":"image_url","image_url":{"url":"https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"}},{"type":"text","text":"Describe this image in one sentence."} }
]

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct", 
	messages=messages, 
	max_tokens=500,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content)