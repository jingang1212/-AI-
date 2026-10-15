"""租户/数据范围：用户可见的企业、站点、设备，以及访问校验。

数据范围 = user_company 关联表（用户 → 企业），站点/设备由企业下钻：
    user → companies → stations → devices
工具层与 REST 层共用这里的校验函数——「接口过滤 + 工具过滤」双保险。

未登录（无 user 上下文）时一律 fail-closed 拒绝，
只有显式声明 all_access 的演示上下文（CLI / 单测）才放开。
"""

from __future__ import annotations

from typing import Any

from . import database


class ScopeDenied(PermissionError):
    """无权访问某数据时抛出；工具层捕获后转成文字回给模型。"""


def visible_company_ids(user: dict[str, Any] | None) -> set[int]:
    if user is None:
        return set()
    rows = database.query(
        "SELECT company_id FROM user_company WHERE user_id = ?", (int(user["id"]),)
    )
    return {int(r["company_id"]) for r in rows}


def visible_station_ids(user: dict[str, Any] | None) -> set[int]:
    companies = visible_company_ids(user)
    if not companies:
        return set()
    placeholders = ",".join("?" * len(companies))
    rows = database.query(
        f"SELECT id FROM h_cems_station WHERE company_id IN ({placeholders})",
        tuple(companies),
    )
    return {int(r["id"]) for r in rows}


def _full_access(user: dict[str, Any] | None) -> bool:
    """演示/测试通道：CLI 与单测显式声明全量访问，网络接口永不使用。"""
    return bool(user and user.get("all_access"))


def can_access_station(user: dict[str, Any] | None, station_id: int) -> bool:
    if _full_access(user):
        return True
    return int(station_id) in visible_station_ids(user)


def can_access_device(user: dict[str, Any] | None, device_db_id: int) -> bool:
    if _full_access(user):
        return True
    rows = database.query(
        "SELECT s.company_id FROM h_cems_station_device sd "
        "JOIN h_cems_station s ON s.id = sd.station_id WHERE sd.device_id = ?",
        (int(device_db_id),),
    )
    if not rows:
        return False
    return int(rows[0]["company_id"]) in visible_company_ids(user)


def require_station(user: dict[str, Any] | None, station_id: int) -> int:
    """站点访问校验，通过返回站点 ID，失败抛 ScopeDenied。"""
    if not can_access_station(user, station_id):
        raise ScopeDenied(f"当前账号无权访问站点 {station_id}")
    return int(station_id)


def require_device(user: dict[str, Any] | None, device_db_id: int) -> int:
    """设备访问校验，通过返回设备主键，失败抛 ScopeDenied。"""
    if not can_access_device(user, device_db_id):
        raise ScopeDenied(f"当前账号无权访问设备 {device_db_id}")
    return int(device_db_id)
