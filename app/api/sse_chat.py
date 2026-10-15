"""SSE 流式对话端点：POST /sse/chat

为什么 Web 端用 SSE？（技术方案 §6.4）
- 浏览器原生 EventSource / fetch 流式就能消费，不用引 WebSocket 库；
- SSE 是 HTTP 单向流，走标准代理/CDN 更顺；
- 微信小程序不支持 SSE/EventSource，所以小程序走 WebSocket（见 ws_chat.py）。

数据格式（统一事件模型，见 docs/技术方案.md §6.1）：
    data: {"type": "content.delta", "delta": "你", "ts": ...}
    每个事件一行 data:，事件之间空行分隔。
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..agents import graph
from ..agents.trace import MAX_QUEUE_SIZE
from .auth import check_token, resolve_role

router = APIRouter()

# 单个请求最长等待时间：诊断链路要跑多轮模型调用，给足余量
REQUEST_TIMEOUT_SECONDS = 180


class ChatBody(BaseModel):
    message: str
    session_id: str = "default"
    role: str = "ops"


@router.post("/sse/chat", dependencies=[Depends(check_token)])
async def sse_chat(
    body: ChatBody,
    x_role: str | None = Header(default=None, alias="X-Role"),
) -> StreamingResponse:
    """以 SSE 流式返回 Agent 的执行过程与最终答案。"""
    role = resolve_role(body.role or x_role)
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)

    # 业务在后台任务里跑，gen() 只负责「从队列取事件 → 序列化 → yield」。
    # 这样模型的输出速率和网络的发送速率解耦：模型快、网络慢也不会卡住业务。
    task = asyncio.create_task(graph.arun(body.message, body.session_id, role, queue=queue))

    async def gen():
        finished = False
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=REQUEST_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    event = {"type": "error", "message": "处理超时，请稍后重试"}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in ("complete", "error"):
                    finished = True
                    break
            # 把后台任务里未转成事件的异常（例如模型调用直接抛错）补一条 error 推出去
            try:
                await task
            except Exception as exc:  # noqa: BLE001
                if not finished:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(exc)[:300]}, ensure_ascii=False)}\n\n"
        finally:
            # 客户端中途断开 → 生成器被关闭 → 走到这里：取消后台任务，避免空转烧 token
            if not task.done():
                task.cancel()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx 等反向代理默认会缓冲响应，SSE 必须关掉（部署时同步改配置）
            "X-Accel-Buffering": "no",
        },
    )
