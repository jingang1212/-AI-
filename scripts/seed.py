"""造演示数据：3 企业、10 站点、30 设备 + 监测数据 + 心跳 + 告警。

注意：本脚本会**先清空再写入**，重复执行不会产生重复数据。
时间口径统一走 app/timeutil.py（本地时区），避免和查询窗口错位 8 小时。
"""

import random
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import database
from app.console import setup_console
from app.timeutil import TS_FORMAT, now

setup_console()


def main():
    database.init_db()
    now_dt = now()

    # 清空既有演示数据，保证脚本可重复执行（幂等）。
    # 顺序：从「引用方」往「被引用方」删，避免外键约束报错。
    for table in ("alarm", "device_heartbeat", "monitor_data", "h_cems_station_device",
                  "h_cems_device", "h_cems_station", "t_company"):
        database.execute(f"DELETE FROM {table}")

    # 3 家企业
    companies = ["华能热电有限公司", "绿源水泥有限公司", "恒昌化工有限公司"]
    for i, name in enumerate(companies, 1):
        database.execute("INSERT INTO t_company (id, name) VALUES (?, ?)", (i, name))

    # 10 个站点
    for i in range(1, 11):
        company_id = (i - 1) % 3 + 1
        database.execute(
            "INSERT INTO h_cems_station (id, company_id, name, station_type, lng, lat) VALUES (?, ?, ?, ?, ?, ?)",
            (i, company_id, f"{i}号排口", 1, 118.3 + i * 0.01, 35.0 + i * 0.01),
        )

    # 30 台设备 + 站点关联（每站点 3 台）
    for i in range(1, 31):
        database.execute(
            "INSERT INTO h_cems_device (id, name, device_type) VALUES (?, ?, ?)",
            (i, f"CEMS-{i:03d}", "烟气CEMS"),
        )
        station_id = (i - 1) % 10 + 1
        database.execute(
            "INSERT INTO h_cems_station_device (id, station_id, device_id) VALUES (?, ?, ?)",
            (i, station_id, i),
        )

    # 监测数据：每站点过去 24 小时每 15 分钟一条 SO2
    for station_id in range(1, 11):
        for step in range(96):
            ts = (now_dt - timedelta(minutes=15 * (96 - step))).strftime(TS_FORMAT)
            value = round(35 + (station_id * 7) % 30 + random.uniform(-3, 3), 1)
            database.execute(
                "INSERT INTO monitor_data (station_id, device_id, ts, metric, value, unit) VALUES (?, ?, ?, ?, ?, ?)",
                (station_id, station_id, ts, "二氧化硫", value, "mg/m³"),
            )

    # 设备心跳
    for device_id in range(1, 31):
        database.execute(
            "INSERT INTO device_heartbeat (device_id, ts, voltage, signal, disk_usage, network_ok, process_running) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (device_id, now_dt.strftime(TS_FORMAT), 12.1, -62, 55.0, 1, 1),
        )

    # 制造两种典型故障，让 Agent 的排障能力有真实素材可查：
    # ① 3 号设备离线 34 分钟（心跳停在过去，电压偏低——疑似断电）
    offline_ts = (now_dt - timedelta(minutes=34)).strftime(TS_FORMAT)
    database.execute("DELETE FROM device_heartbeat WHERE device_id = 3")
    database.execute(
        "INSERT INTO device_heartbeat (device_id, ts, voltage, signal, disk_usage, network_ok, process_running) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (3, offline_ts, 10.4, -78, 91.0, 0, 0),
    )
    # ② 7 号设备在线但采集程序未运行（有心跳、无数据，典型「程序未启动」）
    database.execute(
        "UPDATE device_heartbeat SET process_running = 0 WHERE device_id = 7"
    )

    # 告警：设备级 + 站点级各造几条，用于验证「按设备查告警」和「按站点查告警」两条路径
    alarms = [
        (3, 3, "critical", "设备心跳中断", "待处理"),
        (3, 3, "warning", "二氧化硫小时均值接近限值", "已派单"),
        (7, 7, "warning", "采集程序未运行，数据中断", "待处理"),
        (5, 5, "info", "磁盘使用率超过 85%", "已处理"),
    ]
    for station_id, device_id, level, title, status in alarms:
        database.execute(
            "INSERT INTO alarm (station_id, device_id, ts, level, title, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (station_id, device_id, (now_dt - timedelta(minutes=30)).strftime(TS_FORMAT), level, title, status),
        )
    print("演示数据已写入")


if __name__ == "__main__":
    main()
