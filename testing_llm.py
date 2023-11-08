'''
Going to grab pythia from huggingface so this code will get it and try a simple prompt for a sanity check.
'''

from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m",
)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
print(tokenizer.decode(tokens[0]))


# works!