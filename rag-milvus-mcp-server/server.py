import json
from typing import AsyncIterator, Optional, List
from contextlib import asynccontextmanager
from mcp.server.fastmcp import Context, FastMCP
from milvus import MilvusContext,MilvusConnector,Metadata,Entry
from setting import TOOL_STORE_DESCRIPTION,TOOL_SEARCH_DESCRIPTION

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[MilvusContext]:
    """Milvus 连接 生命周期管理"""
    config = server.config

    connector = MilvusConnector(
        uri=config.get("milvus_uri", "http://localhost:19530"),
        token=config.get("milvus_token"),
        db_name=config.get("db_name", "default"),
    )

    try:
        yield MilvusContext(connector)
    finally:
        pass

mcp = FastMCP("Rag-Mcp-Server", lifespan=server_lifespan)

@mcp.tool(name="milvus-store", description=TOOL_STORE_DESCRIPTION)
async def store(ctx: Context, information: str, metadata: Optional[Metadata] = None,) -> str:
    connector:MilvusConnector = ctx.request_context.lifespan_context.connector
    entry = Entry(content=information, metadata=metadata)
    connector.insert_data(entry)
    return f"Remembered: {information}"


@mcp.tool(name="milvus-search", description=TOOL_SEARCH_DESCRIPTION)
async def search(ctx: Context, query: str, top_n: int = 5, score:float = 0.3) -> List[str]:
    connector:MilvusConnector = ctx.request_context.lifespan_context.connector

    entries = connector.search_collection(query, top_n, score)
    if not entries and len(entries) > 0:
        return [f"No information searched for the query '{query}'"]
    
    content = [f"Results for the query '{query}':"]
    for entry in entries:
        # 生成一个xml格式的结果，让LLM模型更容易解析
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        content.append(f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>")
    return content


