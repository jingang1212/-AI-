"""登录、用户档案与鉴权依赖（JWT）。

生产级设计要点（面试可直接讲）：
- 角色不再由客户端声明：登录成功才签发 JWT，role 从用户表读；
- 每个请求由 get_current_user 解析 Bearer token → 回查数据库，
  以**库里的当前角色**为准（token 里的 role 只作参考），并校验账号未停用；
- 数据范围不放进 JWT（范围会变），而是按需查 user_company 关联表；
- WebSocket 无法带请求头，token 走 ?token= 查询参数（浏览器限制，公认做法）。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Query, WebSocket, status
from pydantic import BaseModel

from .. import database
from ..scope import visible_company_ids
from ..security import decode_token, issue_token, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    username: str
    password: str


def _profile(user: dict) -> dict:
    """用户档案：基本信息 + 可见企业（前端据此渲染，也用于展示权限边界）。"""
    companies = database.query(
        "SELECT c.id, c.name FROM t_company c "
        "JOIN user_company uc ON uc.company_id = c.id WHERE uc.user_id = ? ORDER BY c.id",
        (int(user["id"]),),
    )
    return {
        "id": user["id"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "companies": companies,
    }


@router.post("/login")
def login(body: LoginBody) -> dict:
    rows = database.query("SELECT * FROM users WHERE username = ?", (body.username.strip(),))
    # 用户不存在与密码错误返回同一句话：不暴露「账号是否存在」
    if not rows or not verify_password(body.password, rows[0]["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    user = rows[0]
    if not user["enabled"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已停用")
    return {"token": issue_token(user["id"], user["role"]), "user": _profile(user)}


def _load_user(token: str) -> dict:
    """token → 用户行。任何一步失败都抛 401。"""
    payload = decode_token(token) if token else None
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已失效，请重新登录")
    rows = database.query("SELECT * FROM users WHERE id = ?", (int(payload["sub"]),))
    if not rows or not rows[0]["enabled"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号不存在或已停用")
    return rows[0]


def get_current_user(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> dict:
    """FastAPI 依赖：解析 Bearer token（兼容 ?token=，EventSource 场景）。"""
    raw = token or (authorization or "").removeprefix("Bearer ").strip()
    user = _load_user(raw)
    return {
        "id": user["id"],
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
    }


def get_ws_user(websocket: WebSocket) -> dict | None:
    """WebSocket 连接鉴权：token 走查询参数，失败返回 None（调用方负责关闭连接）。"""
    raw = websocket.query_params.get("token") or ""
    try:
        user = _load_user(raw)
        return {
            "id": user["id"],
            "username": user["username"],
            "name": user["name"],
            "role": user["role"],
        }
    except HTTPException:
        return None


@router.get("/me")
def me(user: dict = Depends(get_current_user)) -> dict:
    return _profile(user)


__all__ = [
    "get_current_user",
    "get_ws_user",
    "router",
    "visible_company_ids",
]
