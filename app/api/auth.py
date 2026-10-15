"""演示用 Bearer Token 鉴权。

说明（面试必被问，提前把话说清楚）：
这是「单一共享 token」的演示方案，token 从 .env 的 AUTH_TOKEN 读取，
校验用 `secrets.compare_digest`（常数时间比较，防时序侧信道）。

生产环境应该升级为：
- 用户体系（登录）→ JWT（HS256/RS256），token 里带 user_id / role / 过期时间；
- 角色不能由请求头传入（那是演示便利），必须从服务端用户信息里查；
- 这就是本项目里 `X-Role` 只用于 demo 的原因——正式版要去掉它。
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, Query, WebSocket, status

from .. import config

# 演示 token：不改默认值就能连，但生产必须改（.env 里配 AUTH_TOKEN）
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "demo-token-please-change")
ALLOWED_ROLES = {"ops", "leader", "service"}


def check_token(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> None:
    """FastAPI 依赖：校验 Bearer token（也支持 ?token= 查询参数，方便 EventSource 直连）。

    设计要点：
    - 支持两种传法是因为浏览器的 EventSource 原生没法带 Authorization 头，
      只能走 query 参数（生产环境改用 fetch + 流式读取，就能带 header 了）；
    - `secrets.compare_digest` 而不是 `==`：即使长度不同也不会提前退出，
      避免通过响应时间猜 token（时序攻击）。
    """
    supplied = token or (authorization or "").removeprefix("Bearer ").strip()
    if not supplied or not secrets.compare_digest(supplied, AUTH_TOKEN):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token 无效")


def check_ws_token(websocket: WebSocket) -> bool:
    """WebSocket 专用校验：连接阶段还没有请求体，token 走 ?token= 查询参数。"""
    supplied = websocket.query_params.get("token") or ""
    return secrets.compare_digest(supplied, AUTH_TOKEN)


def resolve_role(requested: str | None) -> str:
    """把客户端声明的角色收敛为合法角色。

    安全原则：**未知角色按最严格处理（fail-closed）**，落到只有知识检索权限的 service，
    而不是默认给高权限的 ops。攻击者乱填角色只会拿到更少的能力。
    """
    role = (requested or "").strip().lower()
    return role if role in ALLOWED_ROLES else "service"
