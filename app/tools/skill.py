"""技能按需加载（渐进式披露 / progressive disclosure）。

概念：模型不需要把「波特率怎么配、日志怎么导」这些几百行的操作手册
全塞进系统提示词（那样每次请求都白烧 token，还会互相干扰）。
正确做法是——系统提示词里只留一句「有哪些技能可用」，
模型判断需要时再调用 load_skill 把详细手册读进来。

这在主流 Agent 工程里叫 Skills / Tools-as-docs，Claude Skills、
OpenAI 的 tool + retrieval 模式都是这个思路。

官方文档：
- LangChain 工具：https://python.langchain.com/docs/how_to/custom_tools/
- 渐进式披露（Anthropic 工程博客「Building effective agents」里的
  "just-in-time context"）：https://www.anthropic.com/engineering/building-effective-agents
"""

from __future__ import annotations

from langchain_core.tools import tool

from .. import config

SKILL_DIR = config.DATA_DIR / "skills"


def list_skills() -> list[str]:
    """列出当前可用的技能名（不含后缀），供系统提示词动态生成目录。"""
    if not SKILL_DIR.exists():
        return []
    return sorted(p.stem for p in SKILL_DIR.glob("*.md"))


@tool
def load_skill(skill_name: str) -> str:
    """加载某个运维技能的详细操作手册（内容是现场可执行的分步步骤）。
    当问题涉及具体操作（配置波特率、导出日志、更换探头、标定等）时先调用它，
    拿到步骤后再回答用户，不要凭记忆编造操作步骤。

    Args:
        skill_name: 技能名，例如「波特率配置」「故障排查」「设备安装」。
    """
    # 安全点：技能名同样来自模型，必须做白名单校验。
    # 只允许 SKILL_DIR 下真实存在的文件名，杜绝 "../../.env" 这种路径穿越读取敏感文件。
    available = list_skills()
    name = (skill_name or "").strip().removesuffix(".md")
    if name not in available:
        return f"没有名为「{skill_name}」的技能。可用技能：{'、'.join(available) or '（暂无）'}"
    return (SKILL_DIR / f"{name}.md").read_text(encoding="utf-8")
