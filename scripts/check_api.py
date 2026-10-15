"""接口连通性自检：验证 Key 是否有效、模型名是否可用、向量维度是否符合预期。

只依赖 Python 标准库（urllib），所以**不需要先装依赖**就能跑，
用来在 uv sync 之前快速排除「Key 错了 / 模型名写错了」这类低级问题。

用法：
    python scripts/check_api.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Windows 控制台默认是 GBK，直接 print ✅ / 中文会抛 UnicodeEncodeError；
# 强制把标准输出改成 UTF-8，脚本在任何终端都能正常打印。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 —— 某些环境（如重定向到文件）不支持 reconfigure
        pass


def load_env() -> dict[str, str]:
    """读 .env（也允许用系统环境变量覆盖），不依赖 python-dotenv。"""
    env: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    for key in ("DASHSCOPE_API_KEY", "EMBED_MODEL", "CHAT_MODEL", "RERANK_MODEL", "RERANK_URL", "BASE_URL"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def call(url: str, key: str, payload: dict | None = None) -> tuple[int, dict]:
    """发一次请求；返回 (HTTP 状态码, 响应体)。异常也转成结构化结果，不让脚本中断。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "ignore")
        try:
            return exc.code, json.loads(body or "{}")
        except json.JSONDecodeError:
            return exc.code, {"raw": body[:500]}
    except Exception as exc:  # noqa: BLE001 —— 网络问题也要给出人话提示
        return -1, {"error": f"{type(exc).__name__}: {exc}"}


def brief(payload: dict, limit: int = 300) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> None:
    env = load_env()
    key = env.get("DASHSCOPE_API_KEY", "")
    base = env.get("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    embed_model = env.get("EMBED_MODEL", "qwen3.7-text-embedding")
    chat_model = env.get("CHAT_MODEL", "qwen3.8-flash")
    rerank_model = env.get("RERANK_MODEL", "qwen3.7-text-rerank")
    rerank_url = env.get("RERANK_URL", "")

    masked = f"{key[:8]}...{key[-6:]}" if len(key) > 16 else "(空)"
    print(f"Key：{masked}  长度 {len(key)}")
    if not key:
        print("❌ 没有读到 DASHSCOPE_API_KEY，请检查 .env")
        sys.exit(1)

    # ① 对话模型：顺带验证是否支持 Function Calling（Agent 的命根子）
    print(f"\n① 对话模型 {chat_model} …")
    status, body = call(
        f"{base}/chat/completions",
        key,
        {
            "model": chat_model,
            "messages": [{"role": "user", "content": "只回复两个字：正常"}],
            "max_tokens": 16,
        },
    )
    if status == 200:
        content = body.get("choices", [{}])[0].get("message", {}).get("content")
        print(f"   ✅ 可用，模型回复：{content!r}")
    else:
        print(f"   ❌ HTTP {status}：{brief(body)}")

    # ② Function Calling 参数矩阵。
    # 实测坑（qwen3.8-flash）：默认 thinking 模式，此时 tool_choice 不能设 required/object，
    # 会直接 400 «does not support being set to required or object in thinking mode»。
    # 所以这里把四种组合都跑一遍，找出真正能用的写法。
    print(f"\n② Function Calling（工具调用）参数矩阵 …")
    tool_schema = {
        "type": "function",
        "function": {
            "name": "device_status",
            "description": "查询设备状态",
            "parameters": {
                "type": "object",
                "properties": {"device_id": {"type": "string"}},
                "required": ["device_id"],
            },
        },
    }
    combos = [
        ("默认(thinking) + tool_choice=required", {"tool_choice": "required"}),
        ("默认(thinking) + tool_choice=auto", {"tool_choice": "auto"}),
        ("关闭思考 + tool_choice=required", {"tool_choice": "required", "enable_thinking": False}),
        ("关闭思考 + 不传 tool_choice", {"enable_thinking": False}),
    ]
    for title, extra in combos:
        payload = {
            "model": chat_model,
            "messages": [{"role": "user", "content": "帮我查一下 CEMS-003 设备状态"}],
            "tools": [tool_schema],
            **extra,
        }
        status_m, body_m = call(f"{base}/chat/completions", key, payload)
        if status_m == 200:
            calls = body_m.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
            names = [c.get("function", {}).get("name") for c in calls]
            print(f"   ✅ {title}：tool_calls={names}")
        else:
            message = (body_m.get("error") or {}).get("message") or brief(body_m, 120)
            print(f"   ❌ {title}：HTTP {status_m} — {message[:120]}")

    # ③ 向量模型：重点看维度，必须和 Milvus collection 的 dim=1024 一致
    print(f"\n③ 向量模型 {embed_model} …")
    status_emb, body_emb = call(
        f"{base}/embeddings",
        key,
        {"model": embed_model, "input": ["二氧化硫排放限值", "设备离线排查步骤"]},
    )
    if status_emb == 200:
        vectors = [d.get("embedding", []) for d in body_emb.get("data", [])]
        dims = [len(v) for v in vectors]
        print(f"   ✅ 可用，返回 {len(vectors)} 条向量，维度 {dims}")
        if dims and set(dims) != {1024}:
            print("   ⚠️ 维度不是 1024：请同步修改 Milvus collection 的 dim 与方案文档")
    else:
        print(f"   ❌ HTTP {status_emb}：{brief(body_emb)}")

    # ④ rerank 模型（走 DashScope 原生端点，不是 compatible-mode）
    print(f"\n④ rerank 模型 {rerank_model} …")
    status_rr, body_rr = call(
        rerank_url,
        key,
        {
            "model": rerank_model,
            "input": {
                "query": "二氧化硫排放限值是多少",
                "documents": [{"text": "二氧化硫小时均值限值为 50 mg/m³"}, {"text": "设备安装需要接地"}],
            },
            "parameters": {"return_documents": False, "top_n": 2},
        },
    )
    if status_rr == 200:
        print(f"   ✅ 可用：{brief(body_rr.get('output', {}), 260)}")
    else:
        print(f"   ❌ HTTP {status_rr}：{brief(body_rr)}")

    # ⑤ 端点探测：当上面出现 401/403 时，用它判断到底是「Key 不对」还是「地址不对」
    if "--probe" in sys.argv:
        print("\n⑤ 备用端点探测 …")
        probes = [
            (
                "公网可用模型清单（用来看模型名到底该写哪个）",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
            ),
            (
                "公网 dashscope 兼容模式（判断 sk-ws- 这类工作区 Key 能否走公网）",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            ),
            (
                "专属网关 models 列表（判断该 Key 在网关里有没有任何可用模型）",
                f"{base}/models",
            ),
            (
                "专属网关 DashScope 原生文本生成路径",
                f"{base.replace('/compatible-mode/v1', '/api/v1')}/services/aigc/text-generation/generation",
            ),
        ]
        for title, url in probes:
            if url.endswith("/models"):
                status_p, body_p = call(url, key)
                # 模型清单太长，只摘出 id 列表
                ids = [m.get("id") for m in body_p.get("data", [])] if status_p == 200 else []
                if ids:
                    print(f"   · {title}\n     共 {len(ids)} 个：{'、'.join(ids)}")
                    continue
            elif "text-generation" in url:
                status_p, body_p = call(
                    url, key, {"model": chat_model, "input": {"prompt": "你好"}, "parameters": {}}
                )
            else:
                status_p, body_p = call(
                    url, key, {"model": chat_model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
                )
            print(f"   · {title}\n     HTTP {status_p}：{brief(body_p, 240)}")


if __name__ == "__main__":
    main()
