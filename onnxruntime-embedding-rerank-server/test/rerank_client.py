from http_client import HttpClient
import numpy as np
import string

client = HttpClient("http://115.120.51.57:11558/v1")

query = "抗生素是谁发现的？请说明发现过程和时间。"
documents = [
    "路易·巴斯德是微生物学之父，提出了巴氏杀菌法，对疫苗研发有重大贡献。" , # 不相关（无关人物）
    "亚历山大·弗莱明于1928年偶然发现青霉素。他在实验室中发现霉菌抑制了葡萄球菌的生长，这一发现开启了抗生素时代。", # 完全相关（直接回答核心问题）
    "抗生素的滥用可能导致耐药性细菌的产生，世界卫生组织已将其列为全球健康威胁。" , # 弱相关（提及抗生素但未回答发现者）
    "1928年9月28日，弗莱明在伦敦圣玛丽医院记录到青霉素的抑菌效应，但未能纯化该物质。", # 中等相关（提供时间但未完整回答过程）
    "弗莱明因发现青霉素获得1945年诺贝尔生理学或医学奖，但青霉素的大规模生产由霍华德·弗洛里和厄恩斯特·钱恩实现。", # 高度相关（补充关键细节）
]

payload = {
    "model": "Qwen/Qwen3-Reranker-8B",
    "query": query,
    "documents": documents,
    "top_n": 3,  # 可选参数
    "return_documents": True  # 可选参数
}

result = client.post(endpoint="/rerank", data=payload)

print(result)