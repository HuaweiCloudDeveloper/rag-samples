from transformers import AutoTokenizer
import numpy as np

embed_path = "/home/embedding-inference-server/models/bce_embedding_base"
tokenizer = AutoTokenizer.from_pretrained(embed_path)

sentences = ["","Hello"]
             
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="np")

token_num = (inputs['attention_mask'].sum().item() - 2 * inputs['attention_mask'].shape[0])

tokens = (np.sum(inputs["attention_mask"], axis=1) - 2).tolist()

print(f'{inputs},{token_num},{tokens}')

"""
{'input_ids': array([[     0,      6, 124084,      2],[     0,  35378,   6661,      2]]), 'attention_mask': array([[1, 1, 1, 1],[1, 1, 1, 1]])}

token_num = (inputs['attention_mask'].sum().item() - 2 * inputs['attention_mask'].shape[0])
"""