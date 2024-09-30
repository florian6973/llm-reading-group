from transformers import AutoTokenizer, BertForPreTraining
import torch

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.prediction_logits  # https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/bert/modeling_bert.py#L862
seq_relationship_logits = outputs.seq_relationship_logits  # https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/bert/modeling_bert.py#L864


print(prediction_logits.shape, seq_relationship_logits.shape)
print(prediction_logits)
print(seq_relationship_logits)