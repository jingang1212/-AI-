"""FastAPI 应用入口：uvicorn app.api.main:app --port 8000"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .. import config, database, vectorstore
from ..console import setup_console
from . import auth, rest, sse_chat, ws_chat

# Windows 控制台默认 GBK，print ✅/中文 会直接崩（服务器刚踩过）；
# 任何入口（uvicorn / 脚本）进来都先统一输出编码。
setup_console()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动/关闭钩子。

    启动时只做两件事：
    1. 建表（幂等，已有数据不动）；
    2. 体检提示——Key 没配、知识库为空这些「能提前发现的问题」直接打印，
       避免用户连上之后才看到莫名其妙的 500。
    """
    database.init_db()
    if not config.DASHSCOPE_API_KEY:
        print("⚠️ 未配置 DASHSCOPE_API_KEY：对话和知识库都不可用，请检查 .env", file=sys.stderr)
    if not vectorstore.get_all_chunks():
        print("⚠️ 知识库为空：请先执行 python scripts/ingest.py 导入文档", file=sys.stderr)
    token_state = "已自定义" if auth.AUTH_TOKEN != "demo-token-please-change" else "仍是默认值，生产必须改"
    print(f"✅ AI 助手服务已启动（鉴权 token：{token_state}）")
    yield


app = FastAPI(title="环保监测智能运维 AI 助手", version="0.1.0", lifespan=lifespan)

# 演示阶段放开跨域；生产要收紧到具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rest.router, prefix="/api")
app.include_router(sse_chat.router)
app.include_router(ws_chat.router)


@app.get("/healthz")
def healthz() -> dict:
    """健康检查：返回服务与知识库状态，给负载均衡/监控探活用。"""
    return {
        "status": "ok",
        "app": "ai-helper",
        "knowledge_chunks": len(vectorstore.get_all_chunks()),
        "embedding_backend": "dashscope",
    }


# 图表静态目录：draw_trend_chart 生成的 PNG 落在这里，
# 前端拿到 image 事件里的 /charts/xxx.png 后直接 <img> 引用
CHARTS_DIR = config.DATA_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

# 报告下载目录：generate_report 的产物
REPORTS_DIR = config.DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

# 前端页面：单文件聊天页（SSE/WebSocket 双协议、人工确认弹窗都在里面）。
# 注意必须最后挂载 "/"：FastAPI 按注册顺序匹配，先注册的路由优先级更高。
FRONTEND_DIR = config.BASE_DIR / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
