"""主管 Agent（Supervisor）：理解意图 → 派活给员工 → 汇总答案。

主管**不直接调用任何业务工具**，它只会调三个「员工工具」。
这一点很容易在面试时被追问，理由有三条：
1. 降维：主管要做的决策从「20 个工具选哪个」变成「3 个员工派给谁」，
   选错的概率大幅降低（这是多智能体最实在的收益）；
2. 省上下文：每个员工用完自己的工具后只回一句结论，主管不用看几十条工具原始输出；
3. 权限收口：角色能碰的工具在员工层过滤一次，主管层再过滤一次（员工都没有就不暴露）。

官方文档：
- LangGraph 多智能体 Supervisor：
  https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
- tool_choice="required" 强制模型必须调用工具：
  https://python.langchain.com/docs/how_to/structured_output/#forced-tool-calling
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from .. import tools as tool_registry
from ..llm import get_chat_model
from .staff import build_staff_tools
from .state import AgentState
from .trace import trace_event

# 主管最多调度几轮。每轮 = 一次派活 + 员工执行。
# 走到上限就强制进入收尾，把已有信息汇总给用户，避免「主管和员工互相甩锅」死循环。
MAX_ROUNDS = 6


def build_system_prompt(role: str) -> str:
    """拼主管的系统提示词：身份 + 角色权限 + 技能目录 + 收敛规则。

    技巧点：技能目录（有哪些手册可查）用代码动态生成后写进提示词，
    新增手册不用改提示词——这也叫「提示词与数据解耦」。

    同样地，**员工清单也按当前角色实际可用的员工动态生成**：
    如果提示词里写了三个员工、但工具只绑了两个，
    模型会反复尝试调用那个不存在的员工，然后被报错打断，
    表现为「答非所问 + 反复道歉」——这是提示词和工具不一致的典型症状。
    """
    skills = tool_registry.list_skills()
    skill_line = "、".join(skills) if skills else "（暂无）"
    available = [t.name for t in build_staff_tools(role)]
    staff_lines = [
        f"- {name}：{STAFF_INTRO[name]}" for name in available if name in STAFF_INTRO
    ]
    staff_block = "\n".join(staff_lines) if staff_lines else "（当前角色无可用员工）"
    return (
        "你是环保监测智能运维平台的主管，负责把用户的问题拆解后交给合适的员工处理，"
        "最后用一段话把结果整理给用户。\n\n"
        f"{tool_registry.role_prompt(role)}\n\n"
        f"可查的技能手册：{skill_line}\n\n"
        f"当前可调度的员工（只允许调用以下这几个）：\n{staff_block}\n\n"
        "工作规则：\n"
        "1. 一次可以同时派多个员工（并行执行），互不依赖的任务请尽量一起派；\n"
        "2. 员工返回结果后，如果信息已经足够回答用户，就不要再派活，直接给出最终回答；\n"
        "3. 最终回答用中文、结论先行，涉及数据或条款要写明来源；\n"
        "4. 不确定用户指的是哪个站点/设备时，在回答里说明需要用户补充编号，不要随便挑一个。"
    )


# 员工一句话简介：只在这里写一次，提示词按可用性动态拼装
STAFF_INTRO: dict[str, str] = {
    "knowledge_staff": "查法规、标准、手册、SOP（政策与知识类问题）",
    "ops_staff": "查设备状态、心跳、告警、监测数据，做现场排障（安装运维类问题）",
    "report_staff": "查数据 + 找法规依据 + 生成分析报告（总结汇报类问题）",
}


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """主管节点：一次「决策」，决定派哪些员工（或直接收尾）。"""
    role = state.get("role") or tool_registry.DEFAULT_ROLE
    rounds = int(state.get("rounds", 0))
    staff_tools = build_staff_tools(role)

    llm = get_chat_model()
    if staff_tools:
        # 第一轮强制调用员工：用户问了问题，主管不给答案就先派活，
        # 否则模型很容易凭自身知识直接回答（那一瞬间 RAG、工具、权限全都没用上，
        # 在环保监管这种要留痕的场景里是致命的）。
        tool_choice = "required" if rounds == 0 else "auto"
        bound = llm.bind_tools(staff_tools, tool_choice=tool_choice)
    else:
        bound = llm  # 理论上不会发生：所有角色至少都有 knowledge_staff

    messages = [SystemMessage(build_system_prompt(role)), *state["messages"]]
    try:
        ai: AIMessage = await _astream_merged(bound, messages)
    except Exception:
        if not staff_tools:
            raise
        # 兜底：部分模型/网关不一定支持 tool_choice="required"，
        # 报错就退回默认行为（auto），宁可少一次强制派活，也不能让整个请求 500。
        ai = await _astream_merged(llm.bind_tools(staff_tools), messages)

    calls = getattr(ai, "tool_calls", None) or []
    for call in calls:
        # 前端借此展示「正在调度 ops_staff…」的进度气泡
        trace_event("dispatch", staff=call["name"], args=call.get("args"))

    return {"messages": [ai], "rounds": rounds + 1, "task_calls": calls}


async def _astream_merged(bound, messages: list[Any]) -> AIMessage:
    """用流式方式调用模型：把 token 增量推出去，同时拼回完整 AIMessage。

    为什么不自己写合并？（实测踩坑，面试必讲）
    DashScope 流式返回里，工具调用参数是**分片的 JSON 字符串**：
        {"task": "用户      ← 第一个分片
        （安装/运维          ← 后续分片
        工程师）反馈：...
    我第一版手写了「按 id 分组 + 拼字符串」的合并，结果踩了两个坑：
    1. 有的分片 id 是空字符串，被误当成新工具调用，生成了 name 为空的幽灵调用，
       主管连续 6 轮「派活失败 → 重试 → 道歉」；
    2. LangChain 会把部分分片提前 json.loads 成 dict，按 key 覆盖而不是拼接，
       任务文本被截断成两个字。

    正确做法：直接复用 LangChain 为流式聚合设计的 `AIMessageChunk + AIMessageChunk`
    加法合并（官方就是用它做 stream 聚合的），参数分片会按 index 拼接成完整 JSON。
    探针脚本验证过：合并结果完整还原了整段 task。
    """
    merged = None
    async for chunk in bound.astream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            # 打字机推送：每收到一个 token 就发一条 content.delta
            trace_event("content.delta", delta=chunk.content)
        # 关键：用 chunk 加法而不是自己拼。chunk 类型是 AIMessageChunk，
        # 它实现了 __add__，内部会按 index 合并 tool_call 分片并最终解析出完整 args。
        merged = chunk if merged is None else merged + chunk

    content = merged.content if merged is not None else ""
    if isinstance(content, list):
        # 多模态模型会把内容拆成 content blocks；本项目只取文本块
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    tool_calls = getattr(merged, "tool_calls", None) if merged is not None else None
    # 注意：tool_calls 为 None 时不能显式传 None——
    # pydantic 2 校验要求该字段要么是合法列表，要么不传（用默认值）。
    # 显式传 None 会抛「Input should be a valid list」，导致已流出的回答作废重试。
    kwargs: dict[str, Any] = {"content": content}
    if tool_calls:
        kwargs["tool_calls"] = tool_calls
    return AIMessage(**kwargs)


def route_after_supervisor(state: AgentState) -> str:
    """条件路由：还有员工任务要执行 → 去 staff 节点；否则 → 结束。

    这里就是 LangGraph 相对「手写 while 循环」的价值所在：
    路由逻辑是显式函数，可以单测、可以画成图、可以加预算熔断。
    """
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    if calls and int(state.get("rounds", 0)) < MAX_ROUNDS:
        return "staff"
    return "finish"
