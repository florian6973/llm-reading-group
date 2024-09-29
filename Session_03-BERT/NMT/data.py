import pandas
import torch
import os

import spacy

from pprint import pprint
import torchtext.datasets as datasets

from model import load_all_vocab, tokenize, yield_tokens, collate_batch, train_custom_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def read_file(path):
    with open(path, 'r') as f:
        return f.readlines()

# dataset = list(zip(read_file(file_en), read_file(file_fr)))
# print(dataset[:10])

def build_vocabulary(spacy_fr, spacy_en):
    def tokenize_fr(text):
        # print(text)
        text = text[0]
        return tokenize(text, spacy_fr)

    def tokenize_en(text):
        text = text[1]
        return tokenize(text, spacy_en)

    file_en = 'datasets/europarl-v7.fr-en.en'
    file_fr = 'datasets/europarl-v7.fr-en.fr'
    dataset = list(zip(read_file(file_fr), read_file(file_en)))

    print("Building French Vocabulary ...")

    vocab_src = build_vocab_from_iterator(
        map(lambda x: tokenize_fr(x), dataset),
        # yield_tokens(dataset, tokenize_fr, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    vocab_tgt = build_vocab_from_iterator(
        map(lambda x: tokenize_en(x), dataset),
        # yield_tokens(dataset, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_tokenizers():
    try:
        spacy_fr = spacy.load("fr_core_news_sm")
    except IOError:
        os.system("python -m spacy download fr_core_news_sm")
        spacy_fr = spacy.load("fr_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_fr, spacy_en

def load_vocab(spacy_fr, spacy_en):
    if not os.path.exists("vocab_euro.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_fr, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab_euro.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab_euro.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt

# train, val, test = datasets.Multi30k(root='.', language_pair=("en", "de"))

def load_all_vocab():
    print("Load tokenizers")
    spacy_fr, spacy_en = (load_tokenizers())
    print("Load vocab")
    vocab_src, vocab_tgt = (load_vocab(spacy_fr, spacy_en))

    return spacy_fr, spacy_en, vocab_src, vocab_tgt


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_fr,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_fr(text):
        # print(text)
        return tokenize(text, spacy_fr)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    file_en = 'datasets/europarl-v7.fr-en.en'
    file_fr = 'datasets/europarl-v7.fr-en.fr'
    dataset = list(zip(read_file(file_fr), read_file(file_en)))

    # split between train and val
    dataset_train, dataset_val = dataset[:-1000], dataset[-1000:]
    print(dataset_train[:10])

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_fr,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter_map = to_map_style_dataset(
        dataset_train
    )  # DistributedSampler needs a dataset len()
    train_sampler = (
        DistributedSampler(train_iter_map) if is_distributed else None
    )
    valid_iter_map = to_map_style_dataset(dataset_val)
    valid_sampler = (
        DistributedSampler(valid_iter_map) if is_distributed else None
    )

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


print("Load tokenizers")
spacy_fr, spacy_en = (load_tokenizers())
print("Load vocab")
vocab_src, vocab_tgt = (load_vocab(spacy_fr, spacy_en))

# print(vocab_src.lookup_token(10))

# exit()

# train, val = create_dataloaders("cuda", 
#                                 vocab_src,
#                                 vocab_tgt,
#                                 spacy_fr,
#                                 spacy_en,
#                                 12000,
#                                 128,
#                                 False)

print("Train")
train_custom_dataset(vocab_src, vocab_tgt, spacy_fr, spacy_en, create_dataloaders)

# for train_i in train: 
#     print(train_i)
#     break


# def tokenize(label, line):
#     return line.split()

# tokens = []
# for label, line in train:
#     tokens += tokenize(label, line)
#     break

# print(tokens)

# yield_tokens(train + val + test, tokenize_de, index=0)