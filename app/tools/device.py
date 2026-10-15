"""设备状态与监测数据工具：Agent 的「眼睛」。

业务背景（这才是和普通 demo 拉开差距的地方）：
设备装在排污企业现场，可能遇到四种典型故障——
1. 断电：心跳里的电压掉到阈值以下，或干脆不发心跳；
2. 断网：设备还在跑，但云端收不到数据（心跳断流 / network_ok=0）；
3. 程序未启动：心跳正常但 process_running=0，数据不再新增；
4. 协议/配置不同（例如波特率配错）：设备在线但解析不出数据，原始报文堆积。
设备每分钟往云数据库写一条心跳，所以「查云端 MySQL」就能反推现场状态，
不需要远程登录到设备——这也是本项目「设备日志先不上 MCP，先查库」的取舍。

官方文档：
- 工具定义与 Pydantic schema：
  https://python.langchain.com/docs/how_to/custom_tools/
- SQLite 参数化查询（防注入）：
  https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders
"""

from __future__ import annotations

from langchain_core.tools import tool

from .. import database
from ..timeutil import hours_ago_str, now, parse

# 心跳超过这个秒数没来，就判定为「离线」。
# 设备每分钟发一次心跳，容忍 3 个周期（3 分钟）的抖动，避免误报。
OFFLINE_AFTER_SECONDS = 180


def _resolve_device_id(device_id: str) -> tuple[int | None, str | None]:
    """把用户/模型给的设备标识统一成数据库主键。

    兼容三种常见写法：3 / CEMS-003 / cems3。
    找不到就返回 (None, 错误说明)，由调用方转成给模型看的文字，
    而不是抛异常——异常会被 Agent 当成「工具坏了」，文字提示能让它自己纠错重试。
    """
    raw = (device_id or "").strip().upper()
    if not raw:
        return None, "设备编号为空"
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None, f"无法从「{device_id}」解析出设备编号，请使用 CEMS-003 这类格式"
    candidate = int(digits)
    rows = database.query("SELECT id, name FROM h_cems_device WHERE id = ?", (candidate,))
    if not rows:
        return None, f"设备 {device_id} 不存在，请确认设备编号"
    return rows[0]["id"], None


@tool
def device_status(device_id: str) -> str:
    """查询一台电子证据鉴定仪（CEMS）的实时状态与最近的告警，用于现场故障排查。
    返回：在线/离线判断、最近心跳时间、电压、信号强度、磁盘使用率、
    网络是否正常、采集程序是否在运行，以及可能的故障原因提示。

    Args:
        device_id: 设备编号，支持 CEMS-003 或 3 两种写法。
    """
    db_id, err = _resolve_device_id(device_id)
    if err:
        return err

    # 参数化占位符 ? 而不是字符串拼接：SQL 注入在这里就被物理堵死了。
    # 这也是面试问「为什么不让模型直接写 SQL」的标准答案之一。
    heartbeats = database.query(
        "SELECT ts, voltage, signal, disk_usage, network_ok, process_running "
        "FROM device_heartbeat WHERE device_id = ? ORDER BY ts DESC LIMIT 1",
        (db_id,),
    )
    # 告警既有挂在设备上的，也有挂在站点上的（例如「站点数据中断」），
    # 所以两条路径都要查：设备维度 + 该设备所属站点的维度。
    # 只查设备维度会漏掉站点级告警，现场排查时表现为「明明有告警但助手说没有」。
    alarms = database.query(
        "SELECT ts, level, title, status, 'device' AS scope FROM alarm WHERE device_id = ? "
        "UNION ALL "
        "SELECT a.ts, a.level, a.title, a.status, 'station' AS scope FROM alarm a "
        "JOIN h_cems_station_device sd ON sd.station_id = a.station_id "
        "WHERE sd.device_id = ? AND a.station_id IS NOT NULL "
        "ORDER BY ts DESC LIMIT 5",
        (db_id, db_id),
    )
    device_name = database.query("SELECT name FROM h_cems_device WHERE id = ?", (db_id,))[0]["name"]

    if not heartbeats:
        return (
            f"设备 {device_name}（ID={db_id}）：云端从未收到心跳。\n"
            "可能原因：设备未上电、SIM 卡/网线未接、服务器地址配置错误。\n"
            "建议：现场确认指示灯与供电；检查设备上报地址是否为云端公网地址。"
        )

    hb = heartbeats[0]
    last_ts = parse(hb["ts"])
    age = (now() - last_ts).total_seconds()
    online = age <= OFFLINE_AFTER_SECONDS

    lines = [
        f"设备：{device_name}（ID={db_id}）",
        f"在线状态：{'在线' if online else '离线（心跳超时）'}",
        f"最近心跳：{hb['ts']}（{int(age)} 秒前）",
        f"供电电压：{hb['voltage']} V",
        f"信号强度：{hb['signal']} dBm",
        f"磁盘使用率：{hb['disk_usage']} %",
        f"网络上报：{'正常' if hb['network_ok'] else '异常'}",
        f"采集程序：{'运行中' if hb['process_running'] else '未运行'}",
    ]

    # 规则化的「故障推理」：把几个字段组合成一句人话结论。
    # 为什么不全都丢给大模型推理？——现场排障要的是稳定可复现的判断，
    # 规则优先、模型兜底，是运维类 Agent 的常见做法（可解释、可写测试）。
    hints: list[str] = []
    if not online:
        hints.append("心跳中断：优先怀疑现场断电或 4G 断网，再检查设备上报地址是否被改。")
    if not hb["network_ok"]:
        hints.append("心跳有上报但网络标志异常：多为 SIM 卡欠费/信号弱，可让现场拍照确认信号灯。")
    if not hb["process_running"]:
        hints.append("采集程序未运行：设备重启后程序可能未自启，需现场执行启动脚本或检查看门狗。")
    if (hb["disk_usage"] or 0) >= 85:
        hints.append("磁盘接近写满：会导致原始数据落盘失败，建议清理本地缓存目录。")
    if (hb["voltage"] or 0) < 11:
        hints.append("电压偏低：可能是 UPS/电源适配器老化，长时间低压会导致主板反复重启。")
    if alarms:
        hint = "；".join(
            f"{a['ts']} [{a['level']}·{'设备' if a['scope'] == 'device' else '站点'}] "
            f"{a['title']}（{a['status']}）"
            for a in alarms
        )
        hints.append(f"最近告警：{hint}")
    if not hints:
        hints.append("各项指标正常，未发现异常。")

    return "\n".join(lines) + "\n\n排查建议：\n- " + "\n- ".join(hints)


@tool
def query_data(station_id: int, metric: str = "二氧化硫", hours: int = 24) -> str:
    """查询某个站点最近若干小时的监测数据（按小时聚合），用于趋势分析。
    返回每小时的平均值、最大值、最小值与单位，以及整体统计。

    Args:
        station_id: 站点 ID（数字），例如 3 表示 3 号排口。
        metric: 指标名称，例如 二氧化硫；默认 二氧化硫。
        hours: 查询最近多少小时，默认 24，最大 168（7 天）。
    """
    hours = max(1, min(int(hours), 168))  # 夹紧范围：防止模型给出 99999 小时把库拖死
    since = hours_ago_str(hours)

    rows = database.query(
        "SELECT ts, value, unit FROM monitor_data "
        "WHERE station_id = ? AND metric = ? AND ts >= ? ORDER BY ts ASC LIMIT 2000",
        (station_id, metric, since),
    )
    if not rows:
        return f"站点 {station_id} 最近 {hours} 小时没有 {metric} 数据，请确认站点/指标名称是否正确。"

    # 按「小时」聚合：模型看 96 条 15 分钟粒度数据毫无意义，
    # 聚合成 24 个点既省 token 又直接对应环保口径的「小时均值」。
    buckets: dict[str, list[float]] = {}
    for r in rows:
        hour = str(r["ts"])[:13]  # "2026-10-15 08"
        buckets.setdefault(hour, []).append(float(r["value"]))

    unit = rows[0]["unit"] or ""
    lines = [f"站点 {station_id} 最近 {hours} 小时 {metric} 小时均值（单位 {unit}）："]
    for hour in sorted(buckets):
        values = buckets[hour]
        lines.append(f"- {hour}:00  均值 {sum(values) / len(values):.1f}  峰 {max(values):.1f}  谷 {min(values):.1f}")

    all_values = [float(r["value"]) for r in rows]
    lines.append(
        f"整体：平均 {sum(all_values) / len(all_values):.1f}，最高 {max(all_values):.1f}，"
        f"最低 {min(all_values):.1f}，样本 {len(all_values)} 条。"
    )
    return "\n".join(lines)
