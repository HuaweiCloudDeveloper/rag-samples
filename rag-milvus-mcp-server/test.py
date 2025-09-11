from embedding import EmbeddingClient,OpenAIEmbeddingClient

client:EmbeddingClient = OpenAIEmbeddingClient("http://192.168.0.160:8081")

if __name__ == "__main__":
    text = ["Hello World","China"]
    res = client.text_embedding(text)
    print(res)
