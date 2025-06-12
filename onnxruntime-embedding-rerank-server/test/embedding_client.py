from http_client import HttpClient

client = HttpClient("http://127.0.0.1:9997/v1")

sentences = ['中国',"Hello World"]

result = client.post("/embeddings", {"model": "bge","input":sentences})

print(result)