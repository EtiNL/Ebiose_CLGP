import torch
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

prompts = ["This is an example prompt.", "another exemple"]
inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)

model = BertModel.from_pretrained('bert-base-uncased')

# Get the hidden states from the model
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, sequence_length, embedding_dim)

# Typically, the first token ([CLS]) is used as the aggregate representation
prompt_embedding = last_hidden_state[:, 0, :]  # shape: (batch_size, embedding_dim)

print(prompt_embedding.shape)


def get_Bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    return tokenizer, model