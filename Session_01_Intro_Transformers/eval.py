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
idx = torch.LongTensor([[0]])
gen = model.beam_search_abs(idx, model.transition_fn, model.score_fn, 1, 300)
# print(gen)
# gen = model.generate(idx, 1000, temperature=0.1)
# gen = model.greedy_search(idx, 1000) # https://medium.com/@fangkuoyu/a-simple-analysis-of-the-repetition-problem-in-text-generation-c4eb696eb543
for i in range(len(gen)):
    print(decode(gen[i].tolist()))