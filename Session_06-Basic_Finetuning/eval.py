# compute perplexity on test dataset

import utils as u
import torch
import math

import numpy as np
# Optional: To ensure all unused tensors are freed
import gc

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, AutoModel

type_datasets = ["ft-instruct",] #"vanilla", "lora", "ft"]
# type_datasets = ["lora"]


for type_dataset in type_datasets:
    tokenizer, model = u.load_model_and_tokenizer(type_dataset)
    datasets = u.get_dataset(tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False because this is causal/next-token prediction, not masked LM
    )

    # Define training arguments (even though we're evaluating here)
    training_args = TrainingArguments(
        output_dir=f"./results_{type_dataset}",
        per_device_eval_batch_size=1,  # Adjust batch size to fit your GPU/CPU memory
        # very important to avoid overflow, but can speed up things
        logging_dir=f"./logs_{type_dataset}",
        report_to="none"
    )

    # Define a Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=datasets['test'],
    )

    # Compute the loss (and other evaluation metrics if needed)
    results = trainer.evaluate()
    print(f"Loss: {torch.tensor(results['eval_loss'])}")

    np.savetxt(f"loss_{type_dataset}.txt", np.array([results['eval_loss']]))
    np.savetxt(f"perplexity_{type_dataset}.txt", np.exp(np.array([results['eval_loss']])))

    # https://discuss.huggingface.co/t/huge-discrepancy-in-perplexity-of-llm-for-trainer-v-s-scratch-implementation/113673/2

    torch.cuda.empty_cache()

    gc.collect()



# https://discuss.huggingface.co/t/bertformaskedlm-s-loss-and-scores-how-the-loss-is-computed/607/4

# def compute_perplexity(model, dataset, batch_size=8):
#     model.eval()
#     losses = []
    
#     # Iterate over batches
#     for i in range(0, len(dataset), batch_size):
#         batch = dataset[i:i + batch_size]
#         # print(batch)
        
#         with torch.no_grad():
#             # Move input tensors to the appropriate device
#             inputs = {k: v for k, v in batch.items()}
#             outputs = model(**inputs['input_ids'].to(model.device))
#             loss = outputs.loss
#             losses.append(loss.item())
    
#     # Calculate the average loss
#     mean_loss = sum(losses) / len(losses)
#     perplexity = math.exp(mean_loss)
    
#     return perplexity

# # Step 5: Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Step 6: Compute perplexity
# perplexity = compute_perplexity(model, datasets['test'])

# print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
