from datasets import load_dataset
import torch
import re
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, AutoModel

def get_dataset():
    dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv'
    # dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'


    dataset = load_dataset("csv", data_files=dataset_path)
    datasets = dataset['train'].train_test_split(test_size=0.1)
    print(datasets)

    return datasets

def load_model_and_tokenizer(type_dataset):
    model_id = "meta-llama/Llama-3.2-3B"
    model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")

    return tokenizer, model

    