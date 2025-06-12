from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

TOOL_STORE_DESCRIPTION = """
将文档存储到Milvus以供后续检索。

输入参数：
1. 'information' 参数应包含自然语言的文档内容，文本长度小于8000。
2. 'metadata' 参数是一个Python字典，键为字符串，可以存储与该文档相关的元数据。
"""

TOOL_SEARCH_DESCRIPTION = """
从Milvus中使用自然语言形式的文本来搜索相似性文档。

输入参数： 
1. 'query' 参数应该描述你正在寻找的内容，文本长度小于8000，该工具将返回最相关的文档。
2. 'top_n' 参数：要返回相似度最高的前n个文档，字段类型是int，默认值：5。
2. 'score' 参数：要返回相似度大于或等于score的文档，字段类型是float，默认值：0.3。
"""

class MilvusSettings(BaseSettings):
    """
    Configuration for Tablestore.
    """

    uri: Optional[str] = Field(default= 'http://localhost:19530',validation_alias="MILVUS_URI")
    user: Optional[str] = Field(default= '',validation_alias="MILVUS_USER")
    password: Optional[str] = Field(default= '',validation_alias="MILVUS_PASSWORD")
    token: Optional[str] = Field(default= '',validation_alias="MILVUS_TOKEN")
    db_name: Optional[str] = Field(default= 'default',validation_alias="MILVUS_DB_NAME")

class RagSettings(BaseSettings):
    collection_name: str = Field(default="rag_mcp_server", validation_alias="RAG_COLLECTION_NAME")
    vector_dim: int = Field(default=768, validation_alias="RAG_VECTOR_DIMENSION")
        