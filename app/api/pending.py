"""人工确认挂起登记表：session_id -> 用户决定。

为什么必须放共享存储（Redis）？——本项目最有说服力的一条架构理由：
SSE/WebSocket 是**长连接**，而「批准/拒绝」走的是另一个 POST 接口。
多副本部署时，用户的聊天连接在副本 A 上挂起等待，确认回执可能被负载均衡
打到副本 B——如果登记表在各自进程内存里，副本 B 根本找不到这个 session，
确认永远超时。放进 Redis 后，任意副本都能「读到我该等的确认、投递我收到的决定」。

本地单机演示用内存实现即可（留空 REDIS_URL），接口完全一致。
"""

from __future__ import annotations

import asyncio
import json

from .. import config


class PendingStore:
    """两个实现共用的接口（依赖倒置：上层只认识这三个方法）。"""

    async def wait_decision(self, session_id: str, timeout: int) -> dict:  # pragma: no cover
        raise NotImplementedError

    async def resolve(self, session_id: str, decision: dict) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def cleanup(self, session_id: str) -> None:  # pragma: no cover
        raise NotImplementedError


class MemoryPending(PendingStore):
    """内存实现：asyncio.Future 挂在进程字典里。"""

    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future] = {}

    async def wait_decision(self, session_id: str, timeout: int) -> dict:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        old = self._futures.get(session_id)
        if old is not None and not old.done():
            # 同一会话重复请求确认：旧的判为拒绝，防止悬挂
            old.set_result({"approved": False, "reason": "被新的确认请求取代"})
        self._futures[session_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return {"approved": False, "reason": "确认超时，按拒绝处理"}
        finally:
            self._futures.pop(session_id, None)

    async def resolve(self, session_id: str, decision: dict) -> bool:
        future = self._futures.pop(session_id, None)
        if future is None or future.done():
            return False
        future.set_result(decision)
        return True

    async def cleanup(self, session_id: str) -> None:
        future = self._futures.pop(session_id, None)
        if future is not None and not future.done():
            future.set_result({"approved": False, "reason": "会话已结束"})


class RedisPending(PendingStore):
    """Redis 实现：等待方 brpop 阻塞，投递方 rpush 投递。

    协议（顺序很重要，串单 bug 就出在这里）：
    1. 等待方先 SET「等待标记 key」（带 TTL），再在「决定列表 key」上 brpop 阻塞。
       ⚠️ 标记和列表必须是两个 key：SET 存的是字符串，BRPOP 要求列表，
       同一个 key 会被 Redis 报 WRONGTYPE（这个 bug 是实测抓出来的）；
    2. 投递方 resolve 时先 EXISTS 检查等待标记，有才往列表 rpush；
       没有等待方直接返回 False——否则决定会滞留在列表里，
       被下一次同 session 的 brpop 当成「新决定」消费掉（很隐蔽的串单 bug）；
    3. 等待方超时退出前 DEL 两个 key，让迟到的 resolve 拿到 False。
    """

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        # protocol=2：兼容 Redis 5/6（redis-py 5 默认 RESP3 握手，老版本不支持）
        self._redis = aioredis.from_url(url, socket_connect_timeout=2, protocol=2)

    @staticmethod
    def _keys(session_id: str) -> tuple[str, str]:
        return (
            f"ai-helper:confirm:{session_id}:waiting",
            f"ai-helper:confirm:{session_id}:list",
        )

    async def wait_decision(self, session_id: str, timeout: int) -> dict:
        wait_key, list_key = self._keys(session_id)
        await self._redis.set(wait_key, "1", ex=timeout + 60)
        item = await self._redis.brpop(list_key, timeout=timeout)
        await self._redis.delete(wait_key, list_key)
        if not item:
            return {"approved": False, "reason": "确认超时，按拒绝处理"}
        return json.loads(item[1])

    async def resolve(self, session_id: str, decision: dict) -> bool:
        wait_key, list_key = self._keys(session_id)
        if not await self._redis.exists(wait_key):
            return False
        await self._redis.rpush(list_key, json.dumps(decision, ensure_ascii=False))
        await self._redis.expire(list_key, 60)
        return True

    async def cleanup(self, session_id: str) -> None:
        wait_key, list_key = self._keys(session_id)
        await self._redis.delete(wait_key, list_key)


_store: PendingStore | None = None


def get_store() -> PendingStore:
    """进程内单例。按 REDIS_URL 选择实现（Redis 连不上退回内存）。"""
    global _store
    if _store is None:
        if config.REDIS_URL:
            _store = RedisPending(config.REDIS_URL)
        else:
            _store = MemoryPending()
    return _store
