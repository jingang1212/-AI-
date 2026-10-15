"""LangGraph 状态图：把主管和员工组装成一个可运行的 Agent 工作流。

图的形状（面试时可以直接画这个）：

        ┌──────────┐   有员工任务    ┌──────────┐
    START → │ supervisor│ ────────────→ │  staff   │
        └────┬─────┘                └────┬─────┘
             │ 没有任务（已收敛）             │
             ▼                          └──→ 回到 supervisor
            END（把最后一条 AIMessage 作为回答）

为什么用 LangGraph 而不是自己写 if/while？
- 状态机可视化、可打断（interrupt）、可持久化（checkpointer）；
- 断点续跑：用户中途关闭页面，重新进来还能从上次的 checkpoint 继续（D5 会用到）；
- 以后要加「人工确认」「重试」「并行分支」都只是加节点/加边，不用重构业务代码。

官方文档：
- StateGraph 快速入门：https://langchain-ai.github.io/langgraph/tutorials/introduction/
- 检查点（checkpointer）与线程：
  https://langchain-ai.github.io/langgraph/concepts/persistence/
- InMemorySaver / MemorySaver：
  https://langchain-ai.github.io/langgraph/reference/checkpoints/
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .. import config, tools as tool_registry
from .staff import build_staff_tools
from .state import AgentState
from .supervisor import route_after_supervisor, supervisor_node
from .trace import start_trace, trace_event

_CHECKPOINTER: MemorySaver | None = None
_GRAPH = None


def _make_checkpointer():
    """构造 LangGraph checkpointer：配了 REDIS_URL 就用 Redis，否则用内存。

    为什么要有外部 checkpointer？（面试必问，本项目的真实答案）
    - MemorySaver 存在**进程内存**里：后端一重启，所有会话记忆清零，
      用户多轮对话聊到一半就「失忆」；水平扩容到多副本时，A 副本的会话
      在 B 副本上不存在，对话状态互相不一致；
    - 换成 Redis 后，thread_id → checkpoint 落在共享存储，任意副本都能续上
      同一个会话，重启也不丢——这就是「断点续跑 + 多副本部署」的基础设施。

    不做降级：REDIS_URL 一旦配置，就必须真的用 Redis；服务启动时 main.py
    会先验证连通性与 RedisJSON 模块，验证不过直接启动失败并报明原因。
    """
    if not config.REDIS_URL:
        return MemorySaver()

    from langgraph.checkpoint.redis import AsyncRedisSaver

    import redis.asyncio as aioredis

    # 传入自定义客户端：带上 protocol=2 兼容 Redis 5/6（redis-py 5 默认 RESP3 握手）
    client = aioredis.from_url(config.REDIS_URL, protocol=2)
    return AsyncRedisSaver(redis_client=client)


def checkpointer_backend() -> str:
    """当前 checkpointer 用的什么后端（healthz 展示用）。"""
    return "redis" if config.REDIS_URL else "memory"


async def staff_node(state: AgentState) -> dict[str, Any]:
    """员工执行节点：把主管这一轮派的任务**并行**执行完。

    并行有两种实现方式，本项目选了 asyncio.gather：
    - asyncio.gather：先到先跑，全部完成后合并，逻辑简单、异常好处理；
    - LangGraph 的 Send API：把每个任务发成图的独立分支，更「图原生」，
      但和 checkpointer 组合时状态合并规则更绕，D3 阶段先用 gather 跑通。
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    role = state.get("role") or tool_registry.DEFAULT_ROLE
    staff_map = {t.name: t for t in build_staff_tools(role)}

    async def run_one(call: dict[str, Any]) -> ToolMessage:
        name = call["name"]
        args = call.get("args") or {}
        tool_obj = staff_map.get(name)
        if tool_obj is None:
            content = f"员工 {name} 在当前角色（{role}）下不可用。"
        else:
            try:
                content = await tool_obj.ainvoke(args)  # 必须 ainvoke，见 staff.py 注释
            except GraphInterrupt:
                # 写工具的人工确认中断必须冒泡到图引擎，否则确认弹窗永远不出现
                raise
            except Exception as exc:  # noqa: BLE001
                # 一个员工崩了不影响其他员工：把错误当成它的「汇报」交回主管，
                # 主管可以选择换个员工重试，或者如实告诉用户当前能力受限。
                content = f"员工 {name} 执行失败：{type(exc).__name__}: {exc}"
        if not isinstance(content, str):
            content = str(content)
        return ToolMessage(content=content, tool_call_id=call["id"], name=name)

    results = await asyncio.gather(*(run_one(c) for c in calls))
    return {"messages": list(results), "task_calls": []}


def build_graph():
    """组装并编译状态图（编译后才能 ainvoke / astream）。"""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("staff", staff_node)

    graph.add_edge(START, "supervisor")
    # 条件边：返回的字符串 key 对应 mapping 里的目标节点
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"staff": "staff", "finish": END},
    )
    # 员工干完活回到主管，由主管决定「继续派人」还是「收尾」
    graph.add_edge("staff", "supervisor")

    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = _make_checkpointer()
    return graph.compile(checkpointer=_CHECKPOINTER)


def get_graph():
    """进程内单例图对象：编译有成本，别每个请求都 compile 一次。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


async def arun(
    question: str,
    thread_id: str,
    role: str,
    queue: asyncio.Queue | None = None,
    decide: Any = None,
) -> dict[str, Any]:
    """跑一次完整对话，返回 {answer, events, rounds}。

    参数：
    - question：用户这句话；
    - thread_id：会话/线程 ID。LangGraph 用它对号入座地存取 checkpoint，
      同一个 thread_id 的多轮对话会自动带上历史（记忆就是这么实现的）；
    - role：用户角色（ops/leader/service），决定提示词与工具白名单；
    - queue：可选。传入后事件会实时写进队列，由 SSE/WS 端点推给前端；
    - decide：可选。`async def decide(payload) -> dict`，在写操作工具触发 interrupt 时
      负责「把确认请求转给前端并等用户决定」。不传则任何 interrupt 都自动拒绝。
    """
    started_at = time.time()
    graph = get_graph()
    events = start_trace(queue)  # 开启轨迹收集，工具内部会用 trace_event 往里写

    # configurable.thread_id 是 LangGraph 约定的字段名；
    # 传错名字（例如写成 session_id）不会报错，但记忆会「静默失效」，
    # 表现为每次对话模型都忘记上文——这是新手最常见的坑。
    config = {"configurable": {"thread_id": thread_id}}
    # 人工确认循环（LangGraph HITL 的标准写法）：
    # graph.ainvoke 因 interrupt 挂起时抛 GraphInterrupt → 发 interrupt 事件给前端 →
    # 等用户决定（decide 回调，可能几十秒）→ 用 Command(resume=决定) 恢复原图继续跑。
    # 注意恢复时传的必须是 Command(resume=...)，而不是原始输入，否则会重新开始。
    inputs: Any = {"messages": [HumanMessage(question)], "role": role, "rounds": 0, "task_calls": []}
    while True:
        try:
            state = await graph.ainvoke(inputs, config=config)
        except GraphInterrupt as interrupted:
            # 兼容旧版/流式场景：部分版本仍以异常形式抛出中断
            payload = interrupted.args[0] if interrupted.args else {}
            trace_event("interrupt", **payload)
            if decide is None:
                # 无确认通道（比如命令行脚本）：一律拒绝，绝不静默放行
                decision: dict[str, Any] = {"approved": False, "reason": "当前通道不支持人工确认"}
            else:
                decision = await decide(payload)
            inputs = Command(resume=decision)
            continue
        except Exception as exc:  # noqa: BLE001 —— 统一转成 error 事件推给前端
            message = str(exc)
            if "MissingApiKeyError" in type(exc).__name__ or "DASHSCOPE_API_KEY" in message:
                message = "服务端未配置大模型 API Key，请运维检查 .env"
            trace_event("error", message=message[:500])
            raise
        # LangGraph 1.2 的 ainvoke 不再抛 GraphInterrupt，而是把中断放进返回状态：
        # state["__interrupt__"] = [Interrupt(value=..., id=...)]
        interrupts = state.get("__interrupt__")
        if not interrupts:
            break
        interrupted_item = interrupts[-1]  # 多中断按顺序恢复，取最外层一个
        payload = getattr(interrupted_item, "value", interrupted_item)
        if not isinstance(payload, dict):
            payload = {"message": str(payload)}
        trace_event("interrupt", **payload)
        if decide is None:
            # 无确认通道（比如命令行脚本）：一律拒绝，绝不静默放行
            decision: dict[str, Any] = {"approved": False, "reason": "当前通道不支持人工确认"}
        else:
            decision = await decide(payload)
        inputs = Command(resume=decision)

    answer = ""
    for msg in reversed(state["messages"]):
        # 从后往前找「最后一条不带工具调用的 AI 消息」= 主管的最终回答。
        # 不能简单取 messages[-1]：如果主管撞上 MAX_ROUNDS 熔断，
        # 最后一条可能仍是带 tool_calls 的 AIMessage（它没有可读的正文）。
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    if not answer.strip():
        # 兜底：熔断或被模型限流时可能真的没有正文。
        # 宁可给用户一句可理解的说明，也不要返回空字符串让前端显示一片空白
        # ——空响应在前端通常表现为「转圈不动」，用户会以为系统坏了。
        answer = "本次分析未能在有限的调度轮次内收敛，请把问题拆小一些再问一次（例如先问设备状态，再问数据分析）。"
    # 用量统计：主管各轮 AIMessage 的 usage + 员工侧的 staff.usage 事件
    input_tokens = 0
    output_tokens = 0
    for msg in state["messages"]:
        if not isinstance(msg, AIMessage):
            continue
        i, o = _usage_of(msg)
        input_tokens += i
        output_tokens += o
    for e in events:
        if e.get("type") == "staff.usage":
            input_tokens += int(e.get("input_tokens") or 0)
            output_tokens += int(e.get("output_tokens") or 0)
    elapsed = round(time.time() - started_at, 2)
    trace_event("usage", input_tokens=input_tokens, output_tokens=output_tokens, seconds=elapsed)
    _append_trace_log(thread_id, role, question, answer, state, events, input_tokens, output_tokens, elapsed)
    trace_event("complete", answer=answer)
    return {"answer": answer, "events": events, "rounds": int(state.get("rounds", 0))}


_TRACE_LOG = config.DATA_DIR / "traces.jsonl"


def _usage_of(msg: AIMessage) -> tuple[int, int]:
    """从一条 AIMessage 提取 (input_tokens, output_tokens)，兼容两种元信息形态。"""
    usage = getattr(msg, "usage_metadata", None) or {}
    i = int(usage.get("input_tokens") or 0)
    o = int(usage.get("output_tokens") or 0)
    if not i and not o:
        token_usage = (getattr(msg, "response_metadata", None) or {}).get("token_usage") or {}
        i = int(token_usage.get("prompt_tokens") or 0)
        o = int(token_usage.get("completion_tokens") or 0)
    return i, o


def _append_trace_log(
    thread_id: str,
    role: str,
    question: str,
    answer: str,
    state: dict[str, Any],
    events: list[dict[str, Any]],
    input_tokens: int,
    output_tokens: int,
    elapsed: float,
) -> None:
    """请求级轨迹落盘（每行一条 JSON）：token 用量、耗时、工具调用序列。

    这是不依赖外部服务的轻量可观测性：数据落 data/traces.jsonl（git 忽略）。
    观测路径不阻塞主链路——写失败只打一条警告，不影响对话。
    """
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thread_id": thread_id,
        "role": role,
        "question": question[:200],
        "answer": answer[:200],
        "rounds": int(state.get("rounds", 0)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": elapsed,
        "tools": [e.get("name") for e in events if e.get("type") == "tool.call"],
    }
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with _TRACE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 轨迹落盘失败（不影响对话）：{exc}", file=sys.stderr)
