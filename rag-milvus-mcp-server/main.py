import argparse
from dotenv import load_dotenv
from server import mcp

def parse_arguments():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Rag-Mcp-Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="sse")

    return parser.parse_args()

if __name__ == "__main__":
    # 加载.env文件
    load_dotenv()
    # 解析命令行参数
    args = parse_arguments()
    mcp.run(transport=args.transport)
    
