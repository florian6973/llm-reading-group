
# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import torch
import re
from transformers import Trainer, TrainingArguments

from transformers import DataCollatorForLanguageModeling
from transformers import pipeline

import argparse

if __name__ == '__main__':
    
    # Step 1: Create a parser
    parser = argparse.ArgumentParser(description="Choose a model type.")

    # Step 2: Add the 'model_type' argument
    parser.add_argument(
        "model_type",
        choices=["ft", "vanilla"],  # Valid choices
        help="Specify the model type: 'ft' for fine-tuned or 'vanilla' for base model."
    )

    # Step 3: Parse the arguments
    args = parser.parse_args()


    # Define model and tokenizer
    model_id = "meta-llama/Llama-3.2-3B" 

    if args.model_type == 'ft':
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuned_model/checkpoint-19800'
        # dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv'
        # #model_id = "meta-llama/Meta-Llama-3-8B-instruct"
    elif args.model_type == 'vanilla':
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'



    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:1")


    # run everything

    pip = pipeline("text-generation", model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda:1", tokenizer=tokenizer)
    text = pip("Hey how are you doing today?", max_new_tokens=2048)
    print(text)
    text = pip("Medical context. Generate a clinical discharge summary.\nName: ", max_new_tokens=2048)
    print(text)
    print(text[0]['generated_text'])

    # issue with batch size
    # see preprocessed note
    # overfitting to dataset