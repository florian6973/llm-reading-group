
import os 

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

from transformers import DataCollatorForLanguageModeling

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset


import utils as u
tokenizer, model = u.load_model_and_tokenizer("vanilla")
tokenized_datasets = u.get_dataset(tokenizer)

folder = './finetuning_pt'
peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)

# https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning

# # LoRA Configuration
# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,  # For LLaMA, use CAUSAL_LM
#     r=8,                            # Low-rank dimension
#     lora_alpha=32,                  # Scaling factor
#     lora_dropout=0.1,               # Dropout probability
#     target_modules=["q_proj", "v_proj"]  # Apply LoRA to these modules
# )

# Apply LoRA to the LLaMA model
lora_model = get_peft_model(model, peft_config)
print(lora_model.print_trainable_parameters())



training_args = TrainingArguments(
    output_dir=folder,
    num_train_epochs=1,           # Adjust based on your needs
    per_device_train_batch_size=1,  # Adjust based on your hardware
    save_steps=5000,             # Adjust based on your preferences
    eval_steps=1000,              # Adjust based on your preferences
    run_name=folder, # W&B syncing is set to `offline` in this directory.
    report_to="none",
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
    lora_model
)
