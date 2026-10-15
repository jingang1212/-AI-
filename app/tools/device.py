"""设备状态与监测数据工具：Agent 的「眼睛」。

业务背景（这才是和普通 demo 拉开差距的地方）：
设备装在排污企业现场，可能遇到四种典型故障——
1. 断电：心跳里的电压掉到阈值以下，或干脆不发心跳；
2. 断网：设备还在跑，但云端收不到数据（心跳断流 / network_ok=0）；
3. 程序未启动：心跳正常但 process_running=0，数据不再新增；
4. 协议/配置不同（例如波特率配错）：设备在线但解析不出数据，原始报文堆积。
设备每分钟往云数据库写一条心跳，所以「查云端 MySQL」就能反推现场状态，
不需要远程登录到设备——这也是「设备日志不上 MCP，先查库」的取舍。

分层说明：
- 查询逻辑在 app/services/device_service.py（结构化 dict，工具层与 REST 共用）；
- 本文件只做一件事：把 dict 格式化成**大模型读起来最舒服的文本**。

官方文档：
- 工具定义与 Pydantic schema：
  https://python.langchain.com/docs/how_to/custom_tools/
"""

from __future__ import annotations

from langchain_core.tools import tool

from ..auth_context import get_current_user
from ..scope import ScopeDenied, require_device, require_station
from ..services import device_service


@tool
def device_status(device_id: str) -> str:
    """查询一台电子证据鉴定仪（CEMS）的实时状态与最近的告警，用于现场故障排查。
    返回：在线/离线判断、最近心跳时间、电压、信号强度、磁盘使用率、
    网络是否正常、采集程序是否在运行，以及可能的故障原因提示。

    Args:
        device_id: 设备编号，支持 CEMS-003 或 3 两种写法。
    """
    detail = device_service.device_status_detail(device_id)
    if not detail["ok"]:
        return detail["error"]

    # 数据范围校验（AI 工具层的最后一道闸）：模型可能随手传一个不在
    # 用户授权范围内的设备编号，必须拦住——不能只靠接口层过滤。
    try:
        require_device(get_current_user(), int(detail["device_id"]))
    except ScopeDenied as exc:
        return str(exc)

    if detail.get("never_heartbeat"):
        return (
            f"设备 {detail['name']}（ID={detail['device_id']}）：云端从未收到心跳。\n"
            "可能原因：设备未上电、SIM 卡/网线未接、服务器地址配置错误。\n"
            "建议：现场确认指示灯与供电；检查设备上报地址是否为云端公网地址。"
        )

    hb = detail["heartbeat"]
    lines = [
        f"设备：{detail['name']}（ID={detail['device_id']}）",
        f"在线状态：{'在线' if detail['online'] else '离线（心跳超时）'}",
        f"最近心跳：{hb['ts']}（{detail['age_seconds']} 秒前）",
        f"供电电压：{hb['voltage']} V",
        f"信号强度：{hb['signal']} dBm",
        f"磁盘使用率：{hb['disk_usage']} %",
        f"网络上报：{'正常' if hb['network_ok'] else '异常'}",
        f"采集程序：{'运行中' if hb['process_running'] else '未运行'}",
    ]
    return "\n".join(lines) + "\n\n排查建议：\n- " + "\n- ".join(detail["hints"])


@tool
def query_data(station_id: int, metric: str = "二氧化硫", hours: int = 24) -> str:
    """查询某个站点最近若干小时的监测数据（按小时聚合），用于趋势分析。
    返回每小时的平均值、最大值、最小值与单位，以及整体统计。

    Args:
        station_id: 站点 ID（数字），例如 3 表示 3 号排口。
        metric: 指标名称，例如 二氧化硫；默认 二氧化硫。
        hours: 查询最近多少小时，默认 24，最大 168（7 天）。
    """
    try:
        station_id = require_station(get_current_user(), station_id)
    except ScopeDenied as exc:
        return str(exc)

    series = device_service.monitor_series(station_id, metric, hours)
    if not series["ok"]:
        return series["error"]

    lines = [
        f"站点 {station_id} 最近 {series['hours']} 小时 {metric} 小时均值（单位 {series['unit']}）："
    ]
    for point in series["hourly"]:
        lines.append(
            f"- {point['hour']}:00  均值 {point['mean']:.1f}  峰 {point['max']:.1f}  谷 {point['min']:.1f}"
        )
    stats = series["stats"]
    lines.append(
        f"整体：平均 {stats['mean']}，最高 {stats['max']}，最低 {stats['min']}，"
        f"样本 {series['sample_count']} 条。"
    )
    return "\n".join(lines)
