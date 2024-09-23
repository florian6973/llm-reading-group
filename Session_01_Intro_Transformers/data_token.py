from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import decoders
import torch
import os

# Sample text data (replace with your actual data)
# corpus = [
#     "This is a simple example.",
#     "We are testing BPE tokenization in PyTorch.",
#     "BPE helps in efficient subword tokenization."
# ]



with open('tiny_shakespeare.txt', 'r') as f:
    text = f.read()

text = text.replace(' ', '</w> ').replace('\n', ' <nl> ')

corpus = [text]

# Initialize the BPE tokenizer
tokenizer = Tokenizer(BPE(end_of_word_suffix="</w>"))
tokenizer.pre_tokenizer = Whitespace()


# Trainer for BPE
trainer = BpeTrainer(vocab_size=250,
                     special_tokens=['</w>', '<nl>'])
                    #  special_tokens=['<nl>', '<whs>'])#special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])

# Train the BPE tokenizer on your dataset (corpus)

if not os.path.exists('tokenizer.json'):
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.save("tokenizer.json")
else:
    tokenizer = tokenizer.from_file("tokenizer.json")

# Add a decoder for better readability of tokens
tokenizer.decoder = decoders.BPEDecoder()

# Example of tokenization
def tokenize_text(text):
    # Encoding: Tokenizes the text
    text = text.replace(' ', '</w> ').replace('\n', ' <nl> ')
    encoding = tokenizer.encode(text)

    return encoding.ids, encoding.tokens

    # tokens = encoding.tokens
    # tokens_with_space = []
    # for i, token in enumerate(tokens):
    #     if i > 0:
    #         tokens_with_space.append("<space>")
    #     tokens_with_space.append(token)
    
    # return encoding.ids, tokens_with_space

def encode(text):
    return tokenize_text(text)[0]

def decode(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.tolist()
    output_text = tokenizer.decode(tensor, skip_special_tokens=False)
    output_text = output_text.replace('<nl>', '\n')
    return output_text

# Testing BPE tokenization
# for text in corpus:
#     token_ids, tokens = tokenize_text(text)
#     print(f"Original Text: {text}")
#     print(f"Token IDs: {token_ids}")
#     print(f"Tokens: {tokens}")
#     print()

def test():
    # Sample: Using the tokenized output in PyTorch
    sample_text = "BPE helps in tokenization.\nIt is very important, King."
    token_ids, tokens = tokenize_text(sample_text)
    print(tokens)

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(token_ids)
    print("PyTorch Tensor:", input_tensor)

    # Output:
    # Untokenized Text: BPE helps in tokenization.
    output_text = decode(input_tensor)
    # output_text = output_text.replace('</w>', ' ')
    print("Output Text:", output_text)