"""员工 Agent：三个「专才」子智能体，被包装成主管可调用的工具。

架构（和公司小智项目 box-agent-server 的 Supervisor 模式一致）：

    Supervisor（主管，只负责理解意图 + 派活 + 汇总）
        ├── knowledge_staff  知识员工：查知识库、加载技能手册
        ├── ops_staff        运维员工：查设备状态、查监测数据、现场排障
        └── report_staff     报告员工：汇总数据 + 法规依据，产出报告

为什么员工本身也要是 Agent，而不是「工具的一层马甲」？
- 一个真实任务往往要串好几步：查设备状态 → 发现离线 → 查 SOP 怎么处理。
  如果只是普通工具，主管得自己一步步串；员工内部消化掉多步调用，
  主管看到的只是「运维员工返回的结论」，上下文干净很多（这是多智能体的核心收益）。
- 同时必须给员工加**迭代上限**，否则员工内部的工具循环会自己转圈烧 token。

官方文档：
- StructuredTool.from_function（把任意函数包装成工具）：
  https://python.langchain.com/docs/how_to/custom_tools/#structuredtool
- bind_tools（让模型看到工具列表）：
  https://python.langchain.com/docs/how_to/function_calling/
- 多智能体 Supervisor 模式：
  https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field

from .. import tools as tool_registry
from ..llm import get_chat_model
from .trace import trace_event

# 员工内部最多跑几轮「模型思考 → 调工具 → 看结果」。
# 4 轮足够覆盖「检索 → 查库 → 再检索」这类真实链路；
# 超过就强制收敛，避免模型陷入重复调用同一个工具的死循环。
MAX_STEPS = 4

class StaffTaskArgs(BaseModel):
    """主管派给员工的任务参数。

    为什么只有一个 task 字符串？
    主管只需要说「人话任务」，具体要用什么参数由员工自己决定——
    参数 schema 越简单，主管越不容易生成非法调用（多 LLM 协作时，
    参数越多，调用失败率越高的现象非常明显）。
    """

    task: str = Field(description="交给该员工的完整任务描述，包含用户原始问题和已知上下文")


# 员工定义表：名字 -> 人设 + 可用工具
# 工具列表会在运行时再与「用户角色的白名单」求交集（见 build_staff_tools）
STAFF_SPECS: dict[str, dict[str, Any]] = {
    "knowledge_staff": {
        "description": (
            "知识问答专员。擅长从企业知识库中检索法规、排放标准、设备手册、运维 SOP，"
            "并能加载具体操作技能手册。适用于：政策口径、标准限值、安装/运维步骤类问题。"
        ),
        "prompt": (
            "你是环保监测企业的知识问答专员。\n"
            "工作方式：先用 knowledge_search 检索知识库；若问题涉及具体操作步骤，"
            "再用 load_skill 读取对应技能手册。\n"
            "硬性要求：所有结论必须能在检索结果里找到依据，回答时标注来源文件名；"
            "检索不到就如实说「知识库暂无相关条款」，绝对不允许凭记忆编造条款号或数值。\n"
            "输出：给主管看的紧凑结论（200 字内），不要客套话。"
        ),
        "tools": ["knowledge_search", "load_skill"],
    },
    "ops_staff": {
        "description": (
            "设备运维专员。擅长查看电子证据鉴定仪的在线状态、心跳、电压、磁盘、"
            "采集程序运行情况，查询站点监测数据，并给出断电/断网/程序未启动/协议不匹配"
            "等现场故障的排查步骤；也能修改设备配置参数（波特率、上报地址、采样周期），"
            "这类写操作会先弹出确认、用户批准后才生效并留操作日志。"
        ),
        "prompt": (
            "你是电子证据鉴定仪的运维专员，服务对象是现场安装/运维工程师。\n"
            "硬规则（违反即为错误回答）：用户明确要求修改设备配置参数（波特率/上报地址/采样周期）时，"
            "本轮第一步必须直接调用 update_device_config 工具，禁止用文字描述操作步骤代替调用，"
            "也禁止先调用 device_status 再自行判断；人工确认弹窗由该工具自动触发，"
            "是否批准由前端用户决定，不归你判断。\n"
            "工作方式：先 device_status 看设备实时状态和告警；需要趋势时用 query_data；"
            "涉及处理步骤时用 knowledge_search 查运维 SOP，必要时 load_skill 读手册。\n"
            "硬性要求：结论只能基于工具返回的数据，不允许猜测设备编号或数值；"
            "给出建议时按「先看什么、再看什么、怎么处理」的顺序写成分步。\n"
            "输出：给主管看的紧凑结论（250 字内），带关键数值。"
        ),
        "tools": ["device_status", "query_data", "knowledge_search", "load_skill",
                  "update_device_config"],
    },
    "report_staff": {
        "description": (
            "数据报告专员。擅长把监测数据聚合结果、法规依据整理成结构化分析报告"
            "（数据概览 / 趋势分析 / 法规依据 / 结论与建议），并调用工具生成报告文件。"
        ),
        "prompt": (
            "你是数据分析报告专员，服务对象是环保局监管人员。\n"
            "工作方式：先 query_data 拿监测数据，再用 knowledge_search 找对应排放标准条款，"
            "最后用 generate_report 把报告落盘。\n"
            "硬性要求：报告中每个数值都要来自 query_data 的真实返回；"
            "引用标准必须写明文件名称；不得使用「大约」「可能超标」这类模糊表述，"
            "要么给出具体数值对比，要么说明数据不足。\n"
            "输出：给主管看的结论（说明报告已生成、关键发现是什么），200 字内。"
        ),
        "tools": ["query_data", "knowledge_search", "generate_report", "draw_trend_chart"],
    },
}


def _resolve_tools(names: list[str], allowed: set[str]) -> list[BaseTool]:
    """把员工声明用的工具名，过滤成「该角色真的有权限」的工具实例。

    双保险：员工规格里写「我想用哪些」，角色白名单说「实际能给哪些」，
    取交集。这样即使以后给员工加工具时忘了配权限，也不会悄悄泄露给低权限角色。
    """
    return [tool_registry.ALL_TOOLS[n] for n in names if n in allowed]


async def run_staff(staff_name: str, task: str, role: str) -> str:
    """执行一个员工的完整任务，返回它的结论文本。

    内部就是一个手写的 ReAct 循环（LangChain 的 create_agent 也是这个原理，
    手写是为了每一步都能插埋点、能加熔断，也方便面试时讲清楚每一行）：
      ① 把「系统人设 + 任务」发给模型；
      ② 模型返回 tool_calls → 执行工具 → 把结果作为 ToolMessage 塞回消息列表；
      ③ 模型不再调工具（给出纯文本）或到达最大步数 → 结束。
    """
    spec = STAFF_SPECS[staff_name]
    allowed = {t.name for t in tool_registry.tools_for_role(role)}
    staff_tools = _resolve_tools(spec["tools"], allowed)

    if not staff_tools:
        # 角色没权限时明确返回原因，而不是返回空字符串——
        # 空字符串会让主管以为「员工查了但没结果」，进而反复重试。
        return f"[{staff_name}] 当前用户角色（{role}）无权使用该员工的工具，任务未执行。"

    base_llm = get_chat_model()
    messages: list[Any] = [
        SystemMessage(spec["prompt"]),
        HumanMessage(task),
    ]
    trace_event("staff.start", staff=staff_name, task=task)

    last_text = ""
    for step in range(MAX_STEPS):
        # 第一步强制调用工具（和主管第一轮同一策略）：
        # 员工默认也喜欢「先说一段话再干活」，对于写操作请求这会让
        # 确认链路永远走不到 update_device_config——所以第一步不允许空谈。
        if step == 0:
            bound = base_llm.bind_tools(staff_tools, tool_choice="required")
            try:
                ai = await bound.ainvoke(messages)
            except Exception:  # noqa: BLE001 —— 网关不支持 required 时退回 auto 重试一次
                ai = await base_llm.bind_tools(staff_tools).ainvoke(messages)
        else:
            bound = base_llm.bind_tools(staff_tools)
            ai = await bound.ainvoke(messages)
        messages.append(ai)
        calls = getattr(ai, "tool_calls", None) or []

        # 没有工具调用 = 模型认为信息够了，这就是最终结论
        if not calls:
            last_text = ai.content if isinstance(ai.content, str) else str(ai.content)
            break

        for call in calls:
            tool_obj = next((t for t in staff_tools if t.name == call["name"]), None)
            trace_event("tool.call", staff=staff_name, name=call["name"], args=call.get("args"))
            if tool_obj is None:
                result = f"工具 {call['name']} 不存在或无权限"
            else:
                try:
                    # 注意：这里必须用 ainvoke。某些工具（例如后面 D5 的人工确认类工具）
                    # 只实现了异步版本，用同步 invoke 会直接抛「No sync implementation」。
                    result = await tool_obj.ainvoke(call.get("args") or {})
                except GraphInterrupt:
                    # 写工具的人工确认中断必须冒泡到图引擎，否则确认链路永远走不到
                    raise
                except Exception as exc:  # noqa: BLE001 —— 工具异常必须被吃掉并回给模型
                    # 关键设计：单个工具失败不能让整个请求崩掉。
                    # 把错误文本当工具结果回给模型，模型往往能自己换参数重试，
                    # 这是 Agent 稳定性的核心（对应小智项目里的 ToolRetryMiddleware）。
                    result = f"工具执行失败：{type(exc).__name__}: {exc}"
            text_result = result if isinstance(result, str) else str(result)
            trace_event("tool.result", staff=staff_name, name=call["name"], output=text_result[:500])
            messages.append(ToolMessage(content=text_result, tool_call_id=call["id"]))
    else:
        # for-else：循环跑满还未 break，说明到了迭代上限
        last_text = "（员工已达到最大工具调用轮数，以上为当前掌握的信息）"

    trace_event("staff.end", staff=staff_name, output=last_text[:500])
    return last_text or "（员工未返回内容）"


# 角色 -> 员工工具列表缓存。工具是「无状态 + 只读人设」的对象，
# 图里每轮都会取一次，缓存掉能省下每次请求重建闭包的开销。
_STAFF_CACHE: dict[str, list[StructuredTool]] = {}


def build_staff_tools(role: str) -> list[StructuredTool]:
    """把员工包装成主管能 bind 的工具。

    这是整套多智能体的关键技巧：**Agent 即工具**。
    主管的 LLM 只看到 3 个工具（三个员工），看不到几十个底层工具，
    既省上下文，也让「谁来干这件事」的决策变得简单。

    ⚠️ 注意 StructuredTool.from_function 这里只传 coroutine，也就是
    这个工具只有异步实现：调用方必须用 ainvoke / astream_events，
    用同步 invoke 会报错（这正是图里员工节点写成 async 的原因）。
    """
    if role in _STAFF_CACHE:
        return _STAFF_CACHE[role]

    allowed = {t.name for t in tool_registry.tools_for_role(role)}
    staff_tools: list[StructuredTool] = []

    for staff_name, spec in STAFF_SPECS.items():
        if not _resolve_tools(spec["tools"], allowed):
            continue  # 该角色一个相关工具都没有，就别把这个员工暴露给主管

        # 默认参数绑定（default arg）是为了避开 Python 闭包延迟绑定陷阱：
        # 如果在循环里直接引用 staff_name，所有闭包最终都会指向最后一个员工。
        async def _entry(task: str, _name: str = staff_name) -> str:
            return await run_staff(_name, task, role)

        staff_tools.append(
            StructuredTool.from_function(
                coroutine=_entry,
                name=staff_name,
                description=spec["description"],
                args_schema=StaffTaskArgs,
            )
        )
    _STAFF_CACHE[role] = staff_tools
    return staff_tools
