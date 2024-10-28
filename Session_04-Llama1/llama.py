from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

model_path = 'openlm-research/open_llama_3b_v2'
# https://huggingface.co/mediocredev/open-llama-3b-v2-instruct
# model_path = 'openlm-research/open_llama_7b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='cuda:2'
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda:2')

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
).to('cpu')
print(tokenizer.decode(generation_output[0]))