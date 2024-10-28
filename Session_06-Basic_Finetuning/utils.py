from datasets import load_dataset
import torch
import re
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import PeftConfig, PeftModel

from preprocess import get_preprocess_function

def get_dataset(tokenizer=None):
    # dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv'
    dataset_path = '/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'


    dataset = load_dataset("csv", data_files=dataset_path)
    datasets = dataset['train'].train_test_split(test_size=0.1)
    # print(datasets)

    if tokenizer is not None:
        datasets = datasets.map(get_preprocess_function(tokenizer), batched=True)

    return datasets


# https://discuss.huggingface.co/t/config-json-is-not-saving-after-finetuning-llama-2/56871/8
# PEFT transformer vs peft library

def load_model_and_tokenizer(type_dataset, add_padding=True):
    model_id = "meta-llama/Llama-3.2-3B"

    if type_dataset == 'vanilla':
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
    elif type_dataset == 'vanilla-instruct':
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B-Instruct'
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B-Instruct'
    elif type_dataset == 'ft':
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuned_model/checkpoint-19800'
    elif type_dataset == 'ft-instruct':
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B-Instruct'
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuning_instruct_full_10_eos/checkpoint-5000'

        # model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuning_instruct_full_10/checkpoint-99000'
    elif type_dataset == 'lora':
        tokenizer_path = '/gpfs/commons/groups/gursoy_lab/fpollet/models/Llama-3.2-1B'
        model_path = '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuning_lora/checkpoint-9900'
        lora_path =  '/gpfs/commons/groups/gursoy_lab/fpollet/Git/llm-reading-group/Session_06-Basic_Finetuning/finetuning_lora/checkpoint-9900'

        # https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968/3

        model = AutoModelForCausalLM.from_pretrained(tokenizer_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if add_padding:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer)) # https://github.com/pytorch/pytorch/issues/121493
        model = PeftModel.from_pretrained(model, lora_path)
        # print(model)
        model = model.merge_and_unload()
        # model_path = config.base_model_name_or_path

        # model = mo.merge_and_unload()



    if type_dataset != 'lora':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")

    # if type_dataset == 'lora':
    #     model = PeftModel.from_pretrained(model, lora_path)

    if add_padding:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer)) # https://github.com/pytorch/pytorch/issues/121493

    return tokenizer, model

def finetune(folder, training_args, trainer_args, tokenized_datasets, tokenizer, model, train=True):
    # training_args = TrainingArguments(
    #     output_dir=folder,
    #     num_train_epochs=1,           # Adjust based on your needs
    #     per_device_train_batch_size=1,  # Adjust based on your hardware
    #     save_steps=5000,             # Adjust based on your preferences
    #     eval_steps=1000,              # Adjust based on your preferences
    #     run_name=folder, # W&B syncing is set to `offline` in this directory.
    #     report_to="none",
    #     prediction_loss_only=True
    #     # Add additional arguments like learning rate, weight decay, etc.
    # )


    # Define a data collator for causal language modeling
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,  # Set to False because it's causal language modeling, not masked language modeling
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        # data_collator=data_collator,
        **trainer_args
        # Add data collator if needed
    )

    if train:
        trainer.train()


        # Save the fine-tuned model
        trainer.save_model(folder)

    return trainer

    # eval_results = trainer.evaluate()
    # print(eval_results)