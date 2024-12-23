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

### Special notes

To run hyperparameter tuning, `python tune.py`.

To change the tokenizer, please comment/uncomment the line `from data_token import encode, decode` in data.py and adapt the vocabulary size in the GPTconfig.

To change the inference algorithm, please choose to uncomment one line amond 21-23 of `eval.py`.

To run the annotated transformer code, please make sure to install the dependencies. To avoid using an old Pytorch version, I edited the `requirements.txt` to specify a recent working setup. There is also a bug with the data loader in the original code, you may need to force uncompression with `tar -xvf mmt16_task1_[train,val,test].tar.gz` and then delete the archive files from the `datasets` subfolder which was automatically created.

## Session 03 (09-30-2024): BERT

After decoding-only and encoder-decoder architectures, we will investigate encoder-only ones.

We will follow closely the second half of the CS224N class.

### Readings

BERT: https://arxiv.org/abs/1810.04805

### Assignments

- Check if you can use the GPUs of your lab to train the annotated transformer on a dataset of your choice for instance 
- BERT additional preparation: presenter, social and clinical impact, peer-reviewer and hacker

### Session planning

- Review the Annotated Transformer performance after training on GPU
- BERT presentations and discussion

### Special notes

You can download the weights of the trained models (de-en, fr-en) on the shared OneDrive. Please put them in the corresponding folder (`session 02` for the de-en example, `session 03/nmt` for the fr-en example). Then, you just need to run `python eval.py` to run inference. To train the model again, run `python model.py` for de-en or `python data.py` for fr-en in the corresponding folders.

For pretraining BERT: `python bert_pretrain.py --config_name roberta-base --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir test-mlm-pretrain --tokenizer_name FacebookAI/roberta-base`.

For BERT finetuning, please see the notebook `Token_Classification.ipynb`.

## Session 04 (10-07-2024)

- Testing OpenLlama non-finetuned and finetuned (instruct) on GSM8k. 
- Testing Eval-LM to run benchmarks `lm_eval --model hf \
>     --model_args pretrained=openlm-research/open_llama_3b_v2 \
>     --tasks hellaswag \
>     --device cuda:2 \
>     --batch_size 8`.

## Session 04 (10-14-2024)

### Readings

Llama 1: https://arxiv.org/abs/2302.13971

### Assignments

Llama 1: presenter, social and clinical impact, peer-reviewer and hacker

### Session planning

- Llama 1 presentations and discussion

## Code formatting

Please use Black (`pip install black`).