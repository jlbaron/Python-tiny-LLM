'''
Going to grab pythia from huggingface so this code will get it and try a simple prompt for a sanity check.
'''

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m",
)
model.load_state_dict(torch.load('tuned_models\\finetuned.bin'))

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-410m",
)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

inputs = tokenizer("Simple 2 layer neural network in PyTorch. I start by importing torch and then defining a class with nn.Module", return_tensors="pt", max_length=512, truncation=True, padding='max_length')
print(inputs)
# with torch.no_grad():
#     print(inputs)
#     tokens = model(inputs).logits
tokens = model.generate(**inputs, max_new_tokens=512)
print(tokens.shape)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))


# currently returns all <|endoftext|> so definitely a bug somewhere