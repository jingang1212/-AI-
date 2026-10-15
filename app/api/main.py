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
    """启动/关闭钩子：启动时做前置校验，问题在启动阶段就报明原因，不降级。"""
    database.init_db()
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY：请把 .env 里的百炼 API Key 配置好再启动")
    if config.REDIS_URL:
        import redis as sync_redis

        probe = sync_redis.from_url(config.REDIS_URL, socket_connect_timeout=2, protocol=2)
        try:
            probe.ping()
            # checkpointer 底层依赖 RedisJSON 模块（只有 Redis Stack 自带）
            probe.execute_command("JSON.SET", "ai-helper:probe:json", "$", '{"ping": 1}')
            probe.delete("ai-helper:probe:json")
        except Exception as exc:  # noqa: BLE001
            probe.close()
            raise RuntimeError(
                f"Redis 校验失败（{exc}）。checkpointer 需要 RedisJSON 模块（Redis Stack）；"
                "如不使用 Redis，请把 .env 里 REDIS_URL 留空（内存模式，重启丢会话）"
            ) from exc
        probe.close()
    if not vectorstore.get_all_chunks():
        print("⚠️ 知识库为空：请先执行 python scripts/ingest.py 导入文档", file=sys.stderr)
    from ..agents.graph import setup_checkpointer

    await setup_checkpointer()  # 启动即建会话表（SQLite），问题在启动期暴露
    from ..security import get_jwt_secret

    secret_from = "环境变量 JWT_SECRET" if config.JWT_SECRET else "自动生成（data/.jwt_secret）"
    _ = get_jwt_secret()  # 确保密钥文件存在
    print(f"✅ AI 助手服务已启动（JWT 签名密钥来源：{secret_from}）")
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
app.include_router(auth.router)
app.include_router(sse_chat.router)
app.include_router(ws_chat.router)


@app.get("/healthz")
def healthz() -> dict:
    """健康检查：返回服务与知识库状态，给负载均衡/监控探活用。"""
    from ..agents.graph import checkpointer_backend

    return {
        "status": "ok",
        "app": "ai-helper",
        "knowledge_chunks": len(vectorstore.get_all_chunks()),
        "embedding_backend": "dashscope",
        "vector_backend": vectorstore.backend_name(),
        "checkpointer": checkpointer_backend(),
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

# 前端页面：React（Vite）构建产物。开发时用 `npm run dev`（自带代理），
# 生产/演示先 `cd frontend && npm install && npm run build` 生成 dist。
# 构建产物存在就托管；不存在则本次以纯 API 模式启动（不阻断后端）。
# 注意必须最后挂载 "/"：FastAPI 按注册顺序匹配，先注册的路由优先级更高。
FRONTEND_DIR = config.BASE_DIR / "frontend" / "dist"
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    print(
        "⚠️ 未找到前端构建产物（frontend/dist），本次以纯 API 模式启动；"
        "需要页面时执行 cd frontend && npm install && npm run build 后重启",
        file=sys.stderr,
    )
