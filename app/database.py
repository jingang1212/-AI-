"""演示数据层：SQLite 实现。生产切 MySQL/ClickHouse 时接口不变。

表结构与技术方案 §3.1 一致（企业/站点/设备/监测数据/心跳/告警）。
"""

from __future__ import annotations

import sqlite3

from . import config

DB_PATH = config.DATA_DIR / "app.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS t_company (
            id INTEGER PRIMARY KEY, name TEXT, address TEXT
        );
        CREATE TABLE IF NOT EXISTS h_cems_station (
            id INTEGER PRIMARY KEY, company_id INTEGER, name TEXT,
            station_type INTEGER, lng REAL, lat REAL
        );
        CREATE TABLE IF NOT EXISTS h_cems_device (
            id INTEGER PRIMARY KEY, name TEXT, device_type TEXT
        );
        CREATE TABLE IF NOT EXISTS h_cems_station_device (
            id INTEGER PRIMARY KEY, station_id INTEGER, device_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS monitor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT, station_id INTEGER,
            device_id INTEGER, ts TEXT, metric TEXT, value REAL, unit TEXT
        );
        CREATE TABLE IF NOT EXISTS device_heartbeat (
            id INTEGER PRIMARY KEY AUTOINCREMENT, device_id INTEGER, ts TEXT,
            voltage REAL, signal INTEGER, disk_usage REAL,
            network_ok INTEGER, process_running INTEGER
        );
        CREATE TABLE IF NOT EXISTS alarm (
            id INTEGER PRIMARY KEY AUTOINCREMENT, station_id INTEGER,
            device_id INTEGER, ts TEXT, level TEXT, title TEXT, status TEXT
        );
        """
    )
    conn.commit()
    conn.close()


def query(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def execute(sql: str, params: tuple = ()) -> None:
    conn = get_conn()
    conn.execute(sql, params)
    conn.commit()
    conn.close()
