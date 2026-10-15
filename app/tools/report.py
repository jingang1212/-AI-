"""报告生成工具：把数据 + 结论落成 Markdown 文件，供前端下载。

为什么不直接让模型把报告输出到聊天里就完事？
- 报告要能下载、能归档、能盖章（监管场景），必须是文件；
- 聊天窗口塞几千字报告体验极差；
- 落文件后可以继续做「报告列表 / 历史版本 / 区块链存证」——这正是原项目
  电子证据鉴定仪的真实业务闭环（数据 → 报告 → 存证）。
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

from .. import config
from ..agents.trace import trace_event
from ..timeutil import now

REPORT_DIR = config.DATA_DIR / "reports"


def _safe_filename(title: str) -> str:
    """把标题变成安全文件名：去掉 Windows/Linux 非法字符，空标题给默认值。

    安全点：文件名来自模型输出，属于「不可信输入」。
    如果直接拼进路径，模型完全可以生成 "../../xxx"，把文件写到项目外面去
    （路径穿越）。这里只保留中文、字母、数字、下划线、连字符。
    """
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", title or "").strip("_")
    return cleaned[:40] or "report"


@tool
def generate_report(title: str, content: str) -> str:
    """生成分析报告文件（Markdown）。当你已经把数据、依据、结论整理好之后调用它，
    把报告落盘并返回文件路径，前端会提供下载入口。

    Args:
        title: 报告标题，例如「3 号排口二氧化硫 24 小时趋势分析报告」。
        content: 报告正文，Markdown 格式。建议包含「数据概览 / 趋势分析 /
            法规依据 / 结论与建议」四部分，法规依据请引用知识库检索到的原文。
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"{stamp}-{_safe_filename(title)}.md"

    # 头部写死生成时间与工具版本，方便事后追溯——报告是要给人看、给监管看的，
    # 「谁在什么时候用什么版本生成的」比正文本身更容易引起争议。
    header = (
        f"# {title}\n\n"
        f"> 生成时间：{stamp}（{config.TZ_NAME}）\n"
        "> 生成方式：AI 助手自动生成，数据来源为在线监测数据库\n\n"
    )
    path.write_text(header + content, encoding="utf-8")

    # 推事件给前端：渲染「报告下载」链接。
    # 注意只给 URL 和文件名，不给服务器绝对路径（信息泄漏 + 前端也用不上）。
    trace_event("file", url=f"/reports/{path.name}", filename=path.name, title=title)
    # 只把文件名给模型/用户看：绝对路径是服务端实现细节，暴露出去既没用也有信息泄漏风险。
    return f"报告已生成：{path.name}（可在前端「报告下载」中打开）"
