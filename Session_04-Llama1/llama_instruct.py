from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
model_id = 'mediocredev/open-llama-3b-v2-instruct'
tokenizer_id = 'mediocredev/open-llama-3b-v2-instruct'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

pipeline = transformers.pipeline(
  'text-generation',
  model=model_id,
  tokenizer=tokenizer,
  torch_dtype=torch.bfloat16,
  device_map='auto',
)

system_message = 'You are a helpful assistant, who always provide explanation.'
user_message = 'How many days are there in a leap year?'

prompt = f'### System:\n{system_message}<|endoftext|>\n### User:\n{user_message}<|endoftext|>\n### Assistant:\n'
response = pipeline(
   prompt,
   max_length=1000,
   repetition_penalty=1.05,
)
response = response[0]['generated_text']
print(response)

# Assistant: A leap year has 366 days. It's an extra day added to the calendar every four years to account for the extra time it takes for Earth to complete one full orbit around the Sun.
