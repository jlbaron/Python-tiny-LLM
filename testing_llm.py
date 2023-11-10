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
  padding_side='left',
)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

inputs = tokenizer("The following python code sorts 2 lists \n Each list contains int and is no more than 1000 elements long \n We will use numpy operations for extra efficiency: ", return_tensors="pt", max_length=512, truncation=True, padding='max_length')
# with torch.no_grad():
#     tokens = model(inputs).logits
tokens = model.generate(**inputs, max_new_tokens=1024)
print(tokens)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))


# currently returns all <|endoftext|> so definitely a bug somewhere