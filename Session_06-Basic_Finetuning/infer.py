
# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import torch
import re
from transformers import Trainer, TrainingArguments

from transformers import DataCollatorForLanguageModeling
from transformers import pipeline

import argparse

import utils as u

if __name__ == '__main__':
    
    # Step 1: Create a parser
    parser = argparse.ArgumentParser(description="Choose a model type.")

    # Step 2: Add the 'model_type' argument
    parser.add_argument(
        "model_type",
        choices=["ft", "vanilla", "lora", 'ft-instruct'],  # Valid choices
        help="Specify the model type: 'ft' for fine-tuned or 'vanilla' for base model."
    )

    # Step 3: Parse the arguments
    args = parser.parse_args()

    tokenizer, model = u.load_model_and_tokenizer(
        args.model_type
    )

    # Define model and tokenizer
    # model_id = "meta-llama/Llama-3.2-3B" 

    # if args.model_type == 'ft':
    #     tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
    #     model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuned_model/checkpoint-19800'
    #     # dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv'
    #     # #model_id = "meta-llama/Meta-Llama-3-8B-instruct"
    # elif args.model_type == 'vanilla':
    #     model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
    #     tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'



    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:1")

    import numpy as np
    def set_seed(args):
        np.random.seed(args)
        torch.manual_seed(args)
        torch.cuda.manual_seed_all(args)
    # run everything

    # https://huggingface.co/docs/transformers/generation_strategies
    pip = pipeline("text-generation", model=model, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="cuda:2", tokenizer=tokenizer,
                   do_sample=True, num_beams=2, temperature = 1.5, repetition_penalty = 1.5)
    # set_seed(44)#22)#1)
    set_seed(123)#66 88 #22)#1)

    text2 = pip("Medical context. Generate a clinical discharge summary.\nName: ", max_new_tokens=4096)
    print(text2)
    print(text2[0]['generated_text'])
    with open(f'output_{args.model_type}_2.txt', 'w') as f:
        f.write(text2[0]['generated_text'])

    # text3 = pip("### Instruction: generate a clinical note.\n\n### Answer:\n", max_new_tokens=3000)
    # print(text3[0]['generated_text'])

    # text1 = pip("Hey how are you doing today?", max_new_tokens=2048)
    # print(text1)
    # with open(f'output_{args.model_type}_1.txt', 'w') as f:
        # f.write(text1[0]['generated_text'])

    # issue with batch size
    # see preprocessed note
    # overfitting to dataset

    # https://huggingface.co/docs/transformers/generation_strategies 
    # https://huggingface.co/docs/transformers/llm_tutorial
    # GENERATION STRATEGY
    # RECIPES