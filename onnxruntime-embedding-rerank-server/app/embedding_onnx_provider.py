from onnxruntime import InferenceSession
from transformers import XLMRobertaTokenizer
from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
import os

class EmbeddingONNXProvider():
    def __init__(self,embed_path:str,**kwargs):
        embed_model_path = os.path.join(embed_path,"embed.onnx")
        # 判断文件是否存在
        if not os.path.isfile(embed_model_path):
            raise ValueError(f"{embed_model_path} 文件不存在")
        self.workers = kwargs.get('workers', 8)
        self._tokenizer = XLMRobertaTokenizer.from_pretrained(embed_path)
        providers = ['CPUExecutionProvider']
        self._session = InferenceSession(embed_model_path, providers=providers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.workers)
    
    def tokenizer(self,sentences:List[str]):
        inputs = self._tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="np")
        inputs = {k: v for k, v in inputs.items()}
        tokens = (inputs['attention_mask'].sum().item() - 2 * inputs['attention_mask'].shape[0])
        return inputs,tokens
    
    def embedding(self,inputs):
        output_name = self._session.get_outputs()[0].name
        outputs_onnx = self._session.run(output_names=[output_name], input_feed=inputs)
        embedding = outputs_onnx[0][:,0]
        norm_arr = np.linalg.norm(embedding, axis=1, keepdims=True)
        embeddings_normalized = embedding / norm_arr
        return embeddings_normalized.tolist()
    
    def encode(self,sentences:List[str],batch_size:int=200):    
        """批次执行"""
        futures = []
        for k in range(0, len(sentences), batch_size):
            futures.append(self.thread_pool.submit(self.inference,sentences[k:k+batch_size]))
        
        embeddings_list = []
        total_tokens = 0
        for future in futures:
            embeddings,tokens = future.result()
            total_tokens += tokens
            embeddings_list.extend(embeddings)
        return (embeddings_list,total_tokens)
    
    def inference(self,sentences) -> List:
        inputs,tokens = self.tokenizer(sentences)
        embeddings = self.embedding(inputs)
        return (embeddings,tokens)
