# please run train.py before eval

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import GPTConfig, decode
from model import GPT

from data import text

path = 'trained_models/ckpt.pt'
model = GPT(GPTConfig(vocab_size=251))#66))
model.load_state_dict(torch.load(path, weights_only=False)['model'])
model.eval()

# after 10 min of training
output = ""

idx = torch.LongTensor([[0]])

# gen = model.beam_search_abs(idx, model.transition_fn, model.score_fn, 40, 300)
gen = model.generate(idx, 1000, temperature=1) #0.1)
# gen = model.greedy_search(idx, 1000) # https://medium.com/@fangkuoyu/a-simple-analysis-of-the-repetition-problem-in-text-generation-c4eb696eb543

for i in range(len(gen)):
    char = decode(gen[i].tolist())
    print(char, end='')
    output += char

print()
print("Output:", output)
print()

# bert score
from bert_score import score

# cands = [line.strip() for line in output.split('\n') if len(line.strip()) > 0]
# refs = [line.strip() for line in text.split('\n') if len(line.strip()) > 0]
cands = [output]
refs = [text]
P, R, F1 = score(
    cands,
    refs,
    lang='en',
    verbose=True,
)

print(f'P: {P.mean()} R: {R.mean()} F1: {F1.mean()}')