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

inputs = tokenizer("Simple 2 layer neural network in PyTorch ", return_tensors="pt", max_length=2048, truncation=True, padding='max_length')
tokens = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(tokens[0]))


# currently returns all <|endoftext|> so definitely a bug somewhere