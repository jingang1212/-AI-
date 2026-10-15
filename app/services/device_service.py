"""设备与监测数据的结构化查询（工具层和 REST API 共用）。"""

from __future__ import annotations

from typing import Any

from .. import database
from ..timeutil import hours_ago_str, now, parse

# 心跳超过这个秒数没来，判定离线（设备每分钟上报一次，容忍 3 个周期抖动）
OFFLINE_AFTER_SECONDS = 180


def resolve_device_id(device_id: str) -> tuple[int | None, str | None]:
    """把用户/模型给的设备标识统一成数据库主键，兼容 3 / CEMS-003 / cems3。"""
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


def device_status_detail(device_id: str) -> dict[str, Any]:
    """查询一台设备的状态、心跳、告警与故障推理，返回结构化 dict。"""
    db_id, err = resolve_device_id(device_id)
    if err:
        return {"ok": False, "error": err}

    heartbeats = database.query(
        "SELECT ts, voltage, signal, disk_usage, network_ok, process_running "
        "FROM device_heartbeat WHERE device_id = ? ORDER BY ts DESC LIMIT 1",
        (db_id,),
    )
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
        return {
            "ok": True,
            "device_id": db_id,
            "name": device_name,
            "online": False,
            "never_heartbeat": True,
            "alarms": alarms,
            "hints": [
                "云端从未收到心跳：设备可能未上电、SIM 卡/网线未接，或上报地址配置错误",
                "建议现场确认指示灯与供电，并核对设备上报地址是否为云端公网地址",
            ],
        }

    hb = heartbeats[0]
    age = (now() - parse(hb["ts"])).total_seconds()
    online = age <= OFFLINE_AFTER_SECONDS

    hints: list[str] = []
    if not online:
        hints.append("心跳中断：优先怀疑现场断电或 4G 断网，再检查设备上报地址是否被改")
    if not hb["network_ok"]:
        hints.append("心跳有上报但网络标志异常：多为 SIM 卡欠费/信号弱，可让现场拍照确认信号灯")
    if not hb["process_running"]:
        hints.append("采集程序未运行：设备重启后程序可能未自启，需现场执行启动脚本或检查看门狗")
    if (hb["disk_usage"] or 0) >= 85:
        hints.append("磁盘接近写满：会导致原始数据落盘失败，建议清理本地缓存目录")
    if (hb["voltage"] or 0) < 11:
        hints.append("电压偏低：可能是 UPS/电源适配器老化，长时间低压会导致主板反复重启")
    for alarm in alarms:
        hints.append(
            f"告警 {alarm['ts']} [{alarm['level']}·{'设备' if alarm['scope'] == 'device' else '站点'}] "
            f"{alarm['title']}（{alarm['status']}）"
        )
    if not hints:
        hints.append("各项指标正常，未发现异常")

    return {
        "ok": True,
        "device_id": db_id,
        "name": device_name,
        "online": online,
        "age_seconds": int(age),
        "heartbeat": hb,
        "alarms": alarms,
        "hints": hints,
    }


def monitor_series(station_id: int, metric: str = "二氧化硫", hours: int = 24) -> dict[str, Any]:
    """站点监测数据按小时聚合，返回结构化 dict（含原始点，供前端画图）。"""
    hours = max(1, min(int(hours), 168))
    since = hours_ago_str(hours)
    rows = database.query(
        "SELECT ts, value, unit FROM monitor_data "
        "WHERE station_id = ? AND metric = ? AND ts >= ? ORDER BY ts ASC LIMIT 2000",
        (station_id, metric, since),
    )
    if not rows:
        return {"ok": False, "error": f"站点 {station_id} 最近 {hours} 小时没有 {metric} 数据"}

    buckets: dict[str, list[float]] = {}
    for row in rows:
        hour = str(row["ts"])[:13]
        buckets.setdefault(hour, []).append(float(row["value"]))

    values = [float(r["value"]) for r in rows]
    return {
        "ok": True,
        "station_id": station_id,
        "metric": metric,
        "unit": rows[0]["unit"] or "",
        "hours": hours,
        "sample_count": len(rows),
        "stats": {
            "mean": round(sum(values) / len(values), 2),
            "max": round(max(values), 2),
            "min": round(min(values), 2),
        },
        "hourly": [
            {
                "hour": hour,
                "mean": round(sum(v) / len(v), 2),
                "max": round(max(v), 2),
                "min": round(min(v), 2),
            }
            for hour, v in sorted(buckets.items())
        ],
    }
