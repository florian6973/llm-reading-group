# 2024-2025 LLM reading group

This repo contains the coding materials for the reading group of Columbia DBMI.

## Setup

Please install the Python dependencies with `pip install -r requirements.txt` in a dedicated Python environment.

## Session 01 (09-16-2024): Intro to Transformers

### Readings

https://www.youtube.com/watch?v=XfpMkf4rD6E

### Session planning

- Question answering on attention blocks
- Live coding: GPT from stratch
    - To train the model: `python train.py`
    - To generate text: `python eval.py`

## Session 02 (09-23-2024): Neural Machine Translation (NMT)

### Readings

http://nlp.seas.harvard.edu/annotated-transformer/ (https://github.com/harvardnlp/annotated-transformer/)

### Assignment

- Build a nanoGPT to produce Shakespear-like text, based on https://github.com/karpathy/nanoGPT and https://github.com/jbxamora/reversenanogpt

### Session planning

- Debug our nano GPTs
- Compare them using the loss, time and BARTscore
- Introduce a better tokenizer, discuss hyperparameter optimization
- Live implement a neural machine translation example, from the reading

## Session 03 (09-30-2024): BERT

After decoding-only and encoder-decoder architectures, we will investigate encoder-only ones.

We will follow closely the second half of the CS224N class.

### Readings

BERT

### Assignments

TBD

### Session planning

TBD

## Code formatting

Please use Black (`pip install black`).