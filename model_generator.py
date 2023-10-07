import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Generate text based on user input
def generate_text(prompt, max_length=100, num_samples=5):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_samples)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Example usage
input_prompt = "Once upon a time"
generated_text_samples = generate_text(input_prompt, num_samples=5)

# Print generated text samples
for i, sample in enumerate(generated_text_samples):
    print(f"Generated Text {i+1}: {sample}")
