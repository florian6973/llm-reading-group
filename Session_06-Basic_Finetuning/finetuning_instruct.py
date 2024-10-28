# https://colab.research.google.com/drive/1TUa9J2J_1Sj-G7mQHX45fKzZtnW3s1vj?usp=sharing#scrollTo=m2x7uMIFyZI1
# huggingface-cli download meta-llama/Llama-3.2-1B --include "original/*" --local-dir Llama-3.2-1B

# https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset
import torch
import re
from transformers import Trainer, TrainingArguments

from transformers import DataCollatorForLanguageModeling

import utils as u

tokenizer, model = u.load_model_and_tokenizer("vanilla-instruct")
tokenized_datasets = u.get_dataset(tokenizer)

folder = './finetuning_instruct_full_1_all_eos'


training_args = TrainingArguments(
    output_dir=folder,
    num_train_epochs=1,           # Adjust based on your needs
    per_device_train_batch_size=1,  # Adjust based on your hardware
    save_steps=5000,             # Adjust based on your preferences
    eval_steps=1000,              # Adjust based on your preferences
    run_name=folder, # W&B syncing is set to `offline` in this directory.
    report_to="none",
    gradient_accumulation_steps=4,
    prediction_loss_only=True
    # Add additional arguments like learning rate, weight decay, etc.
)


# Define a data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False because it's causal language modeling, not masked language modeling
)


u.finetune(
    folder,
    training_args,
    dict(
        data_collator=data_collator
    ),
    tokenized_datasets,
    tokenizer,
    model
)


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     data_collator=data_collator,
#     # Add data collator if needed
# )

# trainer.train()

# # Save the fine-tuned model
# trainer.save_model(folder)

# eval_results = trainer.evaluate()
# print(eval_results)


# https://github.com/meta-llama/llama-recipes
# https://stackoverflow.com/questions/79021544/removing-strange-special-characters-from-outputs-llama-3-1-model
# https://huggingface.co/docs/transformers/v4.45.2/en/training#train

# /gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning

# CUDA_VISIBLE_DEVICES=0 python finetuning.py