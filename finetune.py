'''
In here I will train the pythia model on python coding tasks
'''
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import yaml
import pandas as pd
import json

parser = argparse.ArgumentParser(description='LLM Code Generation')
parser.add_argument('--config', default='config.yaml')

global args
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.safe_load(f)

for key in config:
    for k, v in config[key].items():
        setattr(args, k, v)


# get model and any other extra utilities

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m",
)
tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-410m",
)
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# dataset object to retrieve code samples efficiently
class CodeDataset(Dataset):
    def __init__(self, data_dir='test'):
        self.data_dir = data_dir
        df = pd.read_csv(f'dataset\\{data_dir}.txt', header=None)
        data = []
        with open('dataset\\test_code.jsonl', 'r') as jsonl_file:
            for line in jsonl_file:
                data.append(json.loads(line))
        
        self.code_samples = []
        for row in data:
            key_value = row['url']
            if key_value in df[0].values:
                self.code_samples.append(row)

    def __len__(self):
        return len(self.code_samples)

    def __getitem__(self, idx):
        code_prompt = self.code_samples[idx]['docstring']
        code_sample = self.code_samples[idx]['function']
        # NOTE: originally had this wrong, prompt is input and sample function is label!!!
        sample = tokenizer.encode(
            code_prompt,
            return_tensors='pt',
            max_length=args.max_len,
            truncation=True,
            padding='max_length'
        )
        label = tokenizer.encode(
            code_sample,
            return_tensors='pt',
            max_length=args.max_len,
            truncation=True,
            padding='max_length'
        )
        return sample.squeeze(0), label.squeeze(0)

# train model on batches of data

dataset = CodeDataset()
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


for param in model.parameters():
    param.requires_grad = False

final_layers = model.gpt_neox.layers[-2:]
for param in final_layers.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(final_layers.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
for epoch in range(args.epochs):
    early_stop = 256
    ctr = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).logits
        outputs = outputs.reshape(-1, outputs.shape[-1])
        labels = labels.reshape(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss.item())
        torch.save(model.state_dict(), "tuned_models\\finetuned.bin")
        ctr += 1
        assert(ctr < early_stop)

# outputs = [batch, max_len, embedding_dim]
# labels = [batch, max_len]
# reshape turns into [batch*max_len, embedding_dim] and [batch*max_len]