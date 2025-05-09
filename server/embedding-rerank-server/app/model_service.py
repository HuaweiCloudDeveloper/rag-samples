from embedding_onnx_provider import EmbeddingONNXProvider
from rerank_onnx_provider import RerankONNXProvider
from typing import List

class EmbeddingService:

    def __init__(self, embed_path, workers=8):
        self.provider = EmbeddingONNXProvider(embed_path,workers = workers)
        self.batch_size=200

    def execute(self, sentences: List[str]):
        return self.provider.encode(sentences,self.batch_size)
    
class RerankService:

    def __init__(self, embed_path, workers=4):
        self.provider = RerankONNXProvider(embed_path,workers = workers)
        self.batch_size=200

    def execute(self, query: str, passages: List[str],top_n:int=5,return_documents:bool=False):
        ids_scores = self.provider.rerank(query,passages,self.batch_size)
        all_pairs = [{'id': id, 'score': score, 'document': passages[id] if return_documents else ""} for entry in ids_scores for id, score in zip(entry['ids'], entry['scores'])]
        sorted_pairs = sorted(all_pairs, key=lambda x: x['score'], reverse=True)
        return sorted_pairs[:top_n]