"""监测数据折线图工具：检索数据库真实数值，用 matplotlib 画图给前端。

注意一个容易混淆的点（面试可能被问）：
**不是让大模型画图**，也不是模型凭空生成趋势。流程是——
1. 调用 monitor_series 从数据库检索该站点该指标的真实监测值；
2. 本工具用 matplotlib 把这些数值画成折线图（小时均值 + 峰谷区间）；
3. 图片落盘到 data/charts，通过 `image` 事件把 URL 推给前端；
4. 前端直接渲染 <img>，同时模型用文字总结趋势。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 无界面后端：服务器没有显示器，必须用 Agg

import matplotlib.pyplot as plt  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from .. import config  # noqa: E402
from ..agents.trace import trace_event  # noqa: E402
from ..auth_context import get_current_user  # noqa: E402
from ..scope import ScopeDenied, require_station  # noqa: E402
from ..services import device_service  # noqa: E402
from ..timeutil import now  # noqa: E402

CHARTS_DIR = config.DATA_DIR / "charts"

# Windows 常用中文字体候选：没有中文字体时标签会变成方块（豆腐块），
# 所以先探测字体，实在没有就退回英文标签——图不能画出来是崩的。
_CJK_FONTS = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]


def _pick_font() -> str | None:
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
        return next((name for name in _CJK_FONTS if name in installed), None)
    except Exception:  # noqa: BLE001
        return None


_FONT = _pick_font()


@tool
def draw_trend_chart(station_id: int, metric: str = "二氧化硫", hours: int = 24) -> str:
    """生成某个站点监测指标最近若干小时的趋势折线图。
    适用于用户要求「看图」「趋势图」或分析报告需要配图时。数据来自数据库真实监测值。

    Args:
        station_id: 站点 ID（数字），例如 3。
        metric: 指标名称，例如 二氧化硫。
        hours: 最近多少小时，默认 24，最大 168。
    """
    try:
        station_id = require_station(get_current_user(), station_id)
    except ScopeDenied as exc:
        return str(exc)

    series = device_service.monitor_series(station_id, metric, hours)
    if not series["ok"]:
        return series["error"]

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now().strftime("%Y%m%d-%H%M%S")
    fname = f"trend_{station_id}_{metric}_{stamp}.png"
    path = CHARTS_DIR / fname

    labels = [p["hour"][11:13] + "时" for p in series["hourly"]]
    means = [p["mean"] for p in series["hourly"]]
    highs = [p["max"] for p in series["hourly"]]
    lows = [p["min"] for p in series["hourly"]]
    unit = series["unit"]

    use_cjk = _FONT is not None
    if use_cjk:
        plt.rcParams["font.sans-serif"] = [_FONT]
    plt.rcParams["axes.unicode_minus"] = False  # 中文环境下负号会乱码，必须关掉

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, means, color="#1f77b4", marker="o", linewidth=1.8, label="小时均值" if use_cjk else "mean")
    ax.fill_between(
        labels, lows, highs, color="#1f77b4", alpha=0.15,
        label="峰谷区间" if use_cjk else "min-max",
    )
    ax.set_title(
        f"站点 {station_id} · {metric} 最近 {hours} 小时趋势" if use_cjk else f"station {station_id} {metric}",
        fontsize=13,
    )
    ax.set_ylabel(f"{metric}（{unit}）" if use_cjk else f"{metric} ({unit})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)  # Agg 后端也要手动 close，否则长期运行会泄漏 figure 内存

    # 图片事件推给前端：SSE/WS 收到 {"type":"image","url":"/charts/xxx.png"}
    trace_event("image", url=f"/charts/{fname}", title=f"站点 {station_id} {metric} 趋势图")
    return f"趋势图已生成并推送前端：/charts/{fname}（共 {len(labels)} 个小时点）"
