import torch

from model import make_model
from data import load_all_vocab

# Load the .pt model
spacy_de, spacy_en, vocab_src, vocab_tgt = load_all_vocab()
model = make_model(len(vocab_src), len(vocab_tgt), N=6)
model.load_state_dict(
    torch.load("euro_final.pt", map_location=torch.device("cpu"))
)

# Check if the model is on GPU and move it to CPU (optional)
# model = model.cpu()

# Function to count the total number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Output the number of parameters
print(f"Total number of trainable parameters: {count_parameters(model)}")
