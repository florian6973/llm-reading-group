from dataclasses import dataclass
import os
import torch

with open('tiny_shakespeare.txt', 'r') as f:
    text = f.read()

@dataclass
class GPTConfig:
    block_size: int = 32
    vocab_size: int = 128 # Our char-level vocab size is 65, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

dataset = 'tiny-shakespeare'
data_dir = os.path.join('data', dataset)

# Here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Vocabulary size:', vocab_size)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # Encoder: Take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l if i in itos]) # Decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # First 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# os.makedirs(data_dir, exist_ok=True)
# np.save(data_dir + '/train.bin', train_data.numpy().astype(np.uint16))
# np.save(data_dir + '/val.bin', val_data.numpy().astype(np.uint16))
# torch.save(train_data.int(), data_dir + '/train.bin')
# torch.save(val_data.int(), data_dir + '/val.bin')

# delete *.bin
# os.remove(data_dir + '/train.bin')
# os.remove(data_dir + '/val.bin')

# rename
# os.rename(data_dir + '/train.bin.npy', data_dir + '/train.bin')
# os.rename(data_dir + '/val.bin.npy', data_dir + '/val.bin')