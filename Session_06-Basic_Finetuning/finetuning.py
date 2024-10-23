# https://colab.research.google.com/drive/1TUa9J2J_1Sj-G7mQHX45fKzZtnW3s1vj?usp=sharing#scrollTo=m2x7uMIFyZI1
# huggingface-cli download meta-llama/Llama-3.2-1B --include "original/*" --local-dir Llama-3.2-1B

# https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing


# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import torch
import re
from transformers import Trainer, TrainingArguments

from transformers import DataCollatorForLanguageModeling

# Define model and tokenizer
model_id = "meta-llama/Llama-3.2-3B"
model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv'
# dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
# #model_id = "meta-llama/Meta-Llama-3-8B-instruct"

folder = './finetuned_model_large'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")

# print(model)

# https://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/tabular_load#csv-files
dataset = load_dataset("csv", data_files=dataset_path)
datasets = dataset['train'].train_test_split(test_size=0.1)
print(datasets)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer)) # https://github.com/pytorch/pytorch/issues/121493

def remove_references(text):
    # Remove text after "REFERENCES"
    reference_start = text.find("REFERENCES")
    if reference_start != -1:
        text = text[:reference_start]
    
    return text

def remove_links(text):
    pattern = r"\[\d+\]|\(http[s]?://\S+\)|www\.\S+|[^a-zA-Z0-9\s]" 
    return re.sub(pattern, "", text)

def remove_special_chars(text):
    pattern = r"[^\w\s.]"  
    return re.sub(pattern, "", text)

def preprocess_text(textso):
    texts = []
    for text in textso:
        text = remove_references(text)
        text = remove_links(text)
        text = text.lower() 
        text = remove_special_chars(text)

        text = re.sub(r'\[\d*\]', '', text)  # Remove square brackets containing numbers
        text = re.sub(r'\[.*?\]', '', text)   # Remove other text between square brackets
        
        # Remove occurrences of "fig"
        text = re.sub(r'\bfig.\b', '', text)
        
        
        # Remove numbers
        text = re.sub(r'\b\d+\b', '', text)  # Remove numbers

        # Remove single characters or numbers in a line
        text = re.sub(r'\b\w\b|\b\d\b', '', text)

        # Filter out lines with only a single character, number, or special character
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 1]  # Filter out lines with length <= 1
        text = '\n'.join(lines)

        texts.append(text)

    return texts


def preprocess_function(examples):
    text = examples['text']
    return tokenizer(
        preprocess_text(text),
        return_tensors='pt',
        max_length=4096,  # Adjust the max length as needed
        truncation=True, padding="max_length",
    )

tokenized_datasets = datasets.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir=folder,
    num_train_epochs=1,           # Adjust based on your needs
    per_device_train_batch_size=1,  # Adjust based on your hardware
    save_steps=5000,             # Adjust based on your preferences
    eval_steps=1000,              # Adjust based on your preferences
    run_name='finetuned_model', # W&B syncing is set to `offline` in this directory.
    report_to="none",
    prediction_loss_only=True
    # Add additional arguments like learning rate, weight decay, etc.
)


# Define a data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False because it's causal language modeling, not masked language modeling
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    # Add data collator if needed
)

trainer.train()

# Save the fine-tuned model
trainer.save_model(folder)

eval_results = trainer.evaluate()
print(eval_results)


# https://github.com/meta-llama/llama-recipes
# https://stackoverflow.com/questions/79021544/removing-strange-special-characters-from-outputs-llama-3-1-model
# https://huggingface.co/docs/transformers/v4.45.2/en/training#train

# /gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning

# CUDA_VISIBLE_DEVICES=0 python finetuning.py