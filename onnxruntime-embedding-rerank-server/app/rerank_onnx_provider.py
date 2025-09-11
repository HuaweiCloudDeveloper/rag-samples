import onnxruntime as ort
from transformers import XLMRobertaTokenizer
from typing import List
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os

def sigmoid(x):
    x = x.astype('float32')
    scores = 1/(1+np.exp(-x))
    scores = np.clip(1.5*(scores-0.5)+0.5, 0, 1)
    return scores

class RerankONNXProvider():

    def __init__(self,rerank_path,**kwargs):
        
        rerank_model_path = os.path.join(rerank_path,"rerank.onnx")
        # 判断文件是否存在
        if not os.path.isfile(rerank_model_path):
            raise ValueError(f"{rerank_model_path} 文件不存在")
        self._tokenizer = XLMRobertaTokenizer.from_pretrained(rerank_path)
        self.sep_id = self._tokenizer.sep_token_id
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)
        self.max_length = kwargs.get('max_length', 512)
        self.workers = kwargs.get('workers', 8)
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(rerank_model_path, providers=providers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.workers)

    def tokenize_preproc(self, query: str, passages: List[str]):

        def _merge_inputs(chunk1_raw, chunk2):
            chunk1 = deepcopy(chunk1_raw)

            chunk1['input_ids'].append(self.sep_id)
            chunk1['input_ids'].extend(chunk2['input_ids'])
            chunk1['input_ids'].append(self.sep_id)

            # 为分隔符添加 attention mask
            chunk1['attention_mask'].append(chunk2['attention_mask'][0])
            chunk1['attention_mask'].extend(chunk2['attention_mask'])
            chunk1['attention_mask'].append(chunk2['attention_mask'][0])

            if 'token_type_ids' in chunk1:
                token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+2)]
                chunk1['token_type_ids'].extend(token_type_ids)
            return chunk1
        
        query_inputs = self._tokenizer(query, truncation=False, padding=False)
        max_passage_inputs_length = self.max_length - len(query_inputs['input_ids']) - 2
        # 校验最大长度
        if max_passage_inputs_length <= 10:
            raise ValueError("Max document length after query is too small.")
        
        overlap_tokens_implt = min(self.overlap_tokens, max_passage_inputs_length  * 2 // 7)

        res_merge_inputs = []
        res_merge_inputs_pids = []
        for pid, psg in enumerate(passages):
            if not psg.strip():
                continue  # 跳过空文档

            passage_inputs = self._tokenizer(psg, truncation=False, padding=False, add_special_tokens=False)
            passage_inputs_length = len(passage_inputs['input_ids'])

            if passage_inputs_length <= max_passage_inputs_length:
                qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)
            else:
                # 分块处理
                start_id = 0
                while start_id < passage_inputs_length:
                    end_id = start_id + max_passage_inputs_length
                    sub_passage_inputs = {k:v[start_id:end_id] for k,v in passage_inputs.items()}
                    start_id = end_id - overlap_tokens_implt if end_id < passage_inputs_length else end_id

                    qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                    res_merge_inputs.append(qp_merge_inputs)
                    res_merge_inputs_pids.append(pid)
        return res_merge_inputs, res_merge_inputs_pids

    def rerank(self, query: str, passages:List[str], batch_size:int=256):
        # passages = [p[:128000] for p in passages if isinstance(p, str) and 0 < len(p)]
        if query is None or len(query) == 0 or len(passages) == 0:
            raise ValueError("query is null or passages is null")
        
        sentence_pairs, sentence_pairs_pids = self.tokenize_preproc(query,passages)

        ids_scores = []
        for k in range(0, len(sentence_pairs), batch_size):
            future = self.thread_pool.submit(self.inference, sentence_pairs[k:k+batch_size])
            ids_scores.append({"ids":sentence_pairs_pids[k:k+batch_size],"scores":future})

        for future in ids_scores:
            future['scores'] = future['scores'].result()
        return ids_scores
    
    def inference(self,sentences):

        def sigmoid(x):
            x = x.astype('float32')
            scores = 1 / (1 + np.exp(-x))
            scores = np.clip(1.5 * (scores - 0.5) + 0.5, 0, 1)
            return scores
        
        # 填充
        inputs = self._tokenizer.pad(sentences,padding=True,max_length=None,pad_to_multiple_of=None,return_tensors=None)
        inputs = {k: v for k, v in inputs.items()}
        result = self.session.run(None, inputs)
        sigmoid_scores = sigmoid(np.array(result[0]))
        return sigmoid_scores.reshape(-1).tolist()