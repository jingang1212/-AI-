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
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .. import tools as tool_registry
from .staff import build_staff_tools
from .state import AgentState
from .supervisor import route_after_supervisor, supervisor_node
from .trace import start_trace, trace_event

_CHECKPOINTER: MemorySaver | None = None
_GRAPH = None


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
        # MemorySaver：进程内内存检查点，重启即丢失。
        # 生产环境必须换成 RedisSaver / PostgresSaver（多副本部署、重启不丢会话），
        # 这也是面试常问的「你的 checkpoint 存哪儿」。
        _CHECKPOINTER = MemorySaver()
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
) -> dict[str, Any]:
    """跑一次完整对话，返回 {answer, events, rounds}。

    参数：
    - question：用户这句话；
    - thread_id：会话/线程 ID。LangGraph 用它对号入座地存取 checkpoint，
      同一个 thread_id 的多轮对话会自动带上历史（记忆就是这么实现的）；
    - role：用户角色（ops/leader/service），决定提示词与工具白名单；
    - queue：可选。传入后事件会实时写进队列，由 SSE/WS 端点推给前端。
    """
    graph = get_graph()
    events = start_trace(queue)  # 开启轨迹收集，工具内部会用 trace_event 往里写

    # configurable.thread_id 是 LangGraph 约定的字段名；
    # 传错名字（例如写成 session_id）不会报错，但记忆会「静默失效」，
    # 表现为每次对话模型都忘记上文——这是新手最常见的坑。
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.ainvoke(
            {"messages": [HumanMessage(question)], "role": role, "rounds": 0, "task_calls": []},
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 —— 统一转成 error 事件推给前端
        message = str(exc)
        if "MissingApiKeyError" in type(exc).__name__ or "DASHSCOPE_API_KEY" in message:
            message = "服务端未配置大模型 API Key，请运维检查 .env"
        trace_event("error", message=message[:500])
        raise

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
    trace_event("complete", answer=answer[:500])
    return {"answer": answer, "events": events, "rounds": int(state.get("rounds", 0))}
