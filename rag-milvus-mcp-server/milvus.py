from pymilvus import DataType, MilvusClient
from pydantic import BaseModel
from pymilvus.milvus_client import IndexParams
from pymilvus.orm.schema import CollectionSchema
from typing import Any, Optional,Dict,List
from setting import MilvusSettings,RagSettings
from embedding import EmbeddingClient

Metadata = Dict[str, Any]

class Entry(BaseModel):

    content: str
    metadata: Optional[Metadata] = None

class MilvusConnector:
    def __init__(self,
                 milvus_settings: MilvusSettings, 
                 embedding_client: EmbeddingClient, 
                 rag_settings: Optional[RagSettings] = None):
        # 创建MilvusClient
        self.client = MilvusClient(uri=milvus_settings.uri,
                                   user=milvus_settings.user, 
                                   password=milvus_settings.password,
                                   token=milvus_settings.token, 
                                   db_name=milvus_settings.db_name)
        
        self.embedding_client = embedding_client
        self.collection_name = rag_settings.collection_name
        if rag_settings and not self.has_collection(rag_settings.collection_name):
            # 定义schema,默认已存在id,vector字段,这里只需要定义document字段
            schema = CollectionSchema()
            schema.add_field("document", DataType.VARCHAR)
            schema.add_field("metadata", DataType.JSON, nullable=True)
            # 定义index
            index_params = IndexParams()
            index_params.add_index("vector","AUTOINDEX","vector_index")
            self.create_collection(collection=rag_settings.collection_name,
                                   dim=rag_settings.vector_dim,
                                   schema= schema,
                                   index_params=index_params
                                   )

    async def has_collection(self,collection_name: str) -> list[str]:
        """查询是否存在collection"""
        try:
            return self.client.has_collection(collection_name)
        except Exception as e:
            raise ValueError(f"Failed to list collections: {str(e)}")
        
    async def create_collection(self, collection_name: str, dim: int = 128, schema: Optional[CollectionSchema] = None, index_params: Optional[IndexParams] = None,) -> bool:
        """从指定的schema创建collection"""
        try:
            # 检查collection是否已存在
            if  self.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' already exists")
            # 创建collection
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                auto_id=True,
                schema=schema,
                index_params = index_params,
                consistency_level = "Strong",
            )
            return True
        except Exception as e:
            raise ValueError(f"Failed to create collection: {str(e)}") 

    async def insert_data(self, entry: Entry) -> dict[str, Any]:
        """插入数据到collection中"""
        try:
            vector = self.embedding_client.text_embedding(entry.content)
            data = {
                "document": entry.content,
                "metadata": entry.metadata,
                "vector": vector
            }
            return self.client.insert(collection_name=self.collection_name, data=data)
        except Exception as e:
            raise ValueError(f"Insert failed: {str(e)}")
        
    async def search_collection(self, query_text: str, top_n: int = 5, score:float = 0.3) -> list[dict]:
        """指定collection进行文本搜索"""
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        vector_data = self.embedding_client.text_embedding(query_text)
        try:
             result:List[List[dict]] = self.client.search(
                collection_name=self.collection_name,
                data=vector_data,
                anns_field="vector",
                search_params=search_params,
                limit=20,
                output_fields=["document"],
            )
        except Exception as e:
            raise ValueError(f"Vector search failed: {str(e)}")

    @classmethod
    async def from_settings(cls,milvus_settings: MilvusSettings,rag_settings: RagSettings):
        return cls(
            
        )

class MilvusContext:
    def __init__(self, connector: MilvusConnector):
        self.connector = connector