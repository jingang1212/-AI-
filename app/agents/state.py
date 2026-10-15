"""LangGraph 全局状态定义。

官方概念（LangGraph 状态图）：
- StateGraph 的每个节点都是 `state -> 局部更新` 的函数，返回值会被「合并」回状态；
- 普通字段是「覆盖」语义（后写覆盖前写）；
- 想「追加」就必须用 reducer 标注，最常用的是 add_messages——
  它会智能合并消息列表：相同 id 的消息覆盖、新的追加，
  并把 HumanMessage/AIMessage/ToolMessage 的对话顺序维护正确。

官方文档：
- 状态与 reducer：https://langchain-ai.github.io/langgraph/concepts/low_level/#state
- add_messages：https://langchain-ai.github.io/langgraph/reference/graphs
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """一次对话请求在图中流转的全部数据。"""

    # 对话消息（含主管的 AIMessage、员工的 ToolMessage）。
    # Annotated[..., add_messages] = 用 add_messages 作为 reducer，做「追加 + 去重」。
    messages: Annotated[list[BaseMessage], add_messages]

    # 用户角色：ops（安装运维）/ leader（环保局领导）/ service（客服）。
    # 角色决定「能用哪些工具」，是权限隔离的输入参数。
    role: str

    # 主管已经调度了几轮（每轮 = 一次 LLM 决策 + 一批员工执行）。
    # 用来做「最大轮数」熔断，防止主管和员工互相踢皮球无限循环。
    rounds: int

    # 本轮主管派发的任务清单（给前端展示「正在调度 XXX 员工」）。
    task_calls: list[dict[str, Any]]
