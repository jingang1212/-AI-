"""全局配置：从 .env 读取。"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3.7-text-embedding")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3.8-flash")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3.7-text-rerank")
RERANK_URL = os.getenv(
    "RERANK_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
)
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
