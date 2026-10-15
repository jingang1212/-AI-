"""工具注册中心：所有 LangChain 工具在这里集中登记，并按用户角色做权限过滤。

为什么要有「注册中心」而不是到处 import？
- 权限隔离需要一个唯一事实来源：哪个角色能碰哪些工具，只在一处定义，
  加新工具时不可能漏配（漏配的默认后果是「没人能用」，而不是「所有人能用」）。
- 面试点：**默认拒绝（default deny）**——白名单里没写的一律不给，
  这比黑名单（先给全部、再逐个禁）安全得多。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from .chart import draw_trend_chart
from .device import device_status, query_data
from .knowledge import knowledge_search
from .report import generate_report
from .skill import list_skills, load_skill
from .write import update_device_config

# 全部工具的名字 -> 实例，员工 Agent 内部按名字取工具用
ALL_TOOLS: dict[str, BaseTool] = {
    t.name: t
    for t in (
        knowledge_search,   # 知识库检索（法规、标准、手册、SOP）
        device_status,      # 设备状态 + 故障推理
        query_data,         # 监测数据聚合查询
        generate_report,    # 报告落盘
        load_skill,         # 技能手册按需加载
        draw_trend_chart,   # 监测数据趋势折线图（数据库真实值）
        update_device_config,  # 设备配置写操作（需人工确认）
    )
}

# 角色 -> 可用工具白名单。
# ops（安装运维）：全套工具，能查设备、查数据、出报告、读手册；
# leader（环保局领导）：只给「看数据 + 出报告」，不给设备侧排障工具——
#   领导不关心波特率，减少工具数量还能降低模型乱调用的概率（工具越多越容易选错）；
# service（客服）：只查知识库，回答政策口径问题，不碰任何业务数据。
ROLE_TOOLS: dict[str, list[str]] = {
    "ops": ["knowledge_search", "device_status", "query_data", "generate_report", "load_skill",
            "draw_trend_chart", "update_device_config"],
    "leader": ["knowledge_search", "query_data", "generate_report", "draw_trend_chart"],
    "service": ["knowledge_search"],
}

# 角色 -> 给主管 Agent 看的「身份说明」，会拼进系统提示词。
# 提示词隔离和工具隔离必须同时做：只隔离工具不隔离提示词，
# 模型会一直承诺自己做不到的事（例如领导问「帮我改波特率」时反复道歉）。
ROLE_PROMPTS: dict[str, str] = {
    "ops": (
        "当前用户是安装/运维工程师：可以查设备状态、查监测数据、查知识库、生成报告、"
        "加载技能手册。回答要给出可现场执行的具体步骤（含命令或操作路径）。"
    ),
    "leader": (
        "当前用户是环保局领导：只需要结论、趋势和法规依据，不要给操作细节。"
        "查询排放数据后要用「是否超标、趋势如何、依据哪条标准」的口径汇报，并可生成分析报告。"
    ),
    "service": (
        "当前用户是客服人员：只依据知识库回答政策、流程类问题；"
        "涉及具体设备或企业数据时，请引导用户提供站点/设备编号。"
    ),
}

DEFAULT_ROLE = "ops"


def tools_for_role(role: str) -> list[BaseTool]:
    """按角色返回可用工具列表；未登记的角色一律只给知识检索（默认拒绝）。"""
    names = ROLE_TOOLS.get(role, ROLE_TOOLS["service"])
    return [ALL_TOOLS[n] for n in names]


def role_prompt(role: str) -> str:
    """返回角色的身份说明文字。"""
    return ROLE_PROMPTS.get(role, ROLE_PROMPTS["service"])


__all__ = [
    "ALL_TOOLS",
    "ROLE_TOOLS",
    "DEFAULT_ROLE",
    "device_status",
    "draw_trend_chart",
    "generate_report",
    "knowledge_search",
    "list_skills",
    "load_skill",
    "query_data",
    "role_prompt",
    "tools_for_role",
    "update_device_config",
]
