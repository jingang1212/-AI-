"""造演示数据：3 企业、10 站点、30 设备 + 监测数据 + 心跳 + 告警。"""

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import database


def main():
    database.init_db()
    now = datetime.now(timezone.utc)

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
            ts = (now - timedelta(minutes=15 * (96 - step))).strftime("%Y-%m-%d %H:%M:%S")
            value = round(35 + (station_id * 7) % 30 + random.uniform(-3, 3), 1)
            database.execute(
                "INSERT INTO monitor_data (station_id, device_id, ts, metric, value, unit) VALUES (?, ?, ?, ?, ?, ?)",
                (station_id, station_id, ts, "二氧化硫", value, "mg/m³"),
            )

    # 设备心跳
    for device_id in range(1, 31):
        database.execute(
            "INSERT INTO device_heartbeat (device_id, ts, voltage, signal, disk_usage, network_ok, process_running) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (device_id, now.strftime("%Y-%m-%d %H:%M:%S"), 12.1, -62, 55.0, 1, 1),
        )

    # 告警
    database.execute(
        "INSERT INTO alarm (station_id, device_id, ts, level, title, status) VALUES (?, ?, ?, ?, ?, ?)",
        (3, 3, now.strftime("%Y-%m-%d %H:%M:%S"), "warning", "二氧化硫小时均值接近限值", "待处理"),
    )
    print("演示数据已写入")


if __name__ == "__main__":
    main()
