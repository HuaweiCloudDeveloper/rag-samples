from typing import List
import torch
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel

class EmbeddingProvider:

    def __init__(self):
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        self.model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")     

    def destroy():
        destroy_model_parallel()     
    
    def embedding(self,documents:List[str]):       
        outputs = self.model.embed(documents)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        scores = (embeddings[:2] @ embeddings[2:].T)
        return scores.tolist()