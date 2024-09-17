# please run train.py before eval

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import GPTConfig, decode
from model import GPT

path = 'trained_models/ckpt.pt'
model = GPT(GPTConfig(vocab_size=66))
model.load_state_dict(torch.load(path, weights_only=False)['model'])
model.eval()

# after 10 min of training
gen = model.generate(torch.LongTensor([[0]]), 1000)
for i in range(len(gen)):
    print(decode(gen[i].tolist()))