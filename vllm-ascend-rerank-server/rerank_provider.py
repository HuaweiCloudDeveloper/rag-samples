import heapq
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm import LLM
from typing import List

class RerankProvider:

    def __init__(self):
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.query_template = "{prefix}<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        self.model = LLM(
            model='Qwen/Qwen3-Reranker-8B',
            task="score",
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
            tensor_parallel_size=1,
            max_model_len=10000, 
            enable_prefix_caching=True, 
            gpu_memory_utilization=0.5)

    def destroy():
        destroy_model_parallel()

    def rerank(self,query:str,documents:List[str],top_n:int,return_documents:bool=False):
        query_in = self.query_template.format(prefix=self.prefix, query=query)
        documents_in = [self.document_template.format(doc=doc, suffix=self.suffix) for doc in documents]
        outputs = self.model.score(query_in,documents_in)

        all_pairs = [
            {'id': index, 'score': output.outputs.score, 'document': documents[index] if return_documents else ""} 
            for index, output in enumerate(outputs)
        ]
            
        if top_n is not None:
            return heapq.nlargest(top_n, all_pairs, key=lambda x: x['score'])
        else:
            return all_pairs