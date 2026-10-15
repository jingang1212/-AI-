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

# 业务时区。设备上报、报表统计、用户提问全部按这个时区对齐，
# 详见 app/timeutil.py 里的说明（时区不一致会导致「查不到数据」的假故障）。
TZ_NAME = os.getenv("TZ_NAME", "Asia/Shanghai")

# Redis 连接串（可空）。本项目两处用到 Redis：
# 1. LangGraph checkpointer：会话记忆 / 断点续跑 / 多副本一致性；
# 2. 人工确认的挂起登记表（SSE 连接与确认回执可能落在不同副本）。
# 留空 = 本地内存模式（重启丢会话）；配置后必须可用（checkpointer 需 RedisJSON），
# 启动时会校验，校验失败直接报错，不降级。
REDIS_URL = os.getenv("REDIS_URL", "")

# JWT 签名密钥（可空：留空会在 data/.jwt_secret 自动生成并持久化，重启不失效）
JWT_SECRET = os.getenv("JWT_SECRET", "")

# 向量库（Milvus）：
#   - 默认嵌入式 Milvus Lite（本地文件，零部署）；数据文件落在 DATA_DIR 下；
#   - 生产换成服务端地址即可，同一套代码：MILVUS_URI=http://127.0.0.1:19530
#   - 托管版（Zilliz Cloud）额外填 MILVUS_TOKEN
MILVUS_URI = os.getenv("MILVUS_URI", str(DATA_DIR / "milvus_lite.db"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "knowledge_chunks")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")

# 是否开启「思考模式」。默认关闭，原因（实测踩坑）：
# qwen3.8-flash 默认开思考，此时传 tool_choice="required" 会直接 400：
# «The tool_choice parameter does not support being set to required or object in thinking mode»。
# Agent 场景需要的是「稳定地调对工具」，不是长篇推理，关掉思考还更快更省 token。
# 需要深度分析时可临时开：ENABLE_THINKING=true
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "false").strip().lower() in {"1", "true", "yes"}


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5

# 向量维度：qwen3.7-text-embedding 实测返回 1024 维。
# 换模型必须同步改这里 + Milvus collection 的 dim，否则写入直接报错。
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
