"""只读 REST 接口：设备列表 / 设备状态 / 监测数据 / 文档上传。

返回结构化 JSON（与 Agent 工具共用 app/services 里的查询逻辑），
保证「页面看到的数据」和「AI 助手说出的数据」永远一致。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from .. import database, rag
from ..agents.trace import trace_event
from ..services import device_service
from . import pending
from .auth import check_token

router = APIRouter(dependencies=[Depends(check_token)])


class UploadResult(BaseModel):
    filename: str
    document_id: str
    chunk_count: int
    strategy: str


class ConfirmBody(BaseModel):
    session_id: str
    approved: bool
    comment: str = ""


@router.post("/chat/confirm")
def confirm_decision(body: ConfirmBody) -> dict:
    """人工确认回执：前端收到 interrupt 事件、用户点完按钮后回调这里。

    只负责「唤醒」等待中的 Agent 会话；真正的写入动作仍然由
    config_service.apply_change 在确认通过后执行（本接口本身不做任何数据库修改）。
    """
    ok = pending.resolve(body.session_id, {"approved": body.approved, "comment": body.comment})
    return {"ok": ok, "message": "已提交" if ok else "该会话当前没有等待确认的操作"}


@router.get("/devices")
def list_devices() -> dict:
    """设备列表（含站点关联），给前端下拉框/设备卡片用。"""
    rows = database.query(
        "SELECT d.id, d.name, d.device_type, s.id AS station_id, s.name AS station_name "
        "FROM h_cems_device d "
        "LEFT JOIN h_cems_station_device sd ON sd.device_id = d.id "
        "LEFT JOIN h_cems_station s ON s.id = sd.station_id "
        "ORDER BY d.id"
    )
    return {"ok": True, "count": len(rows), "devices": rows}


@router.get("/devices/{device_id}/status")
def device_status_api(device_id: str) -> dict:
    """设备状态：与 AI 助手的 device_status 工具完全同源。"""
    return device_service.device_status_detail(device_id)


@router.get("/monitor/data")
def monitor_data_api(station_id: int, metric: str = "二氧化硫", hours: int = 24) -> dict:
    """监测数据：小时聚合 + 原始统计。前端折线图直接用 hourly 字段画。"""
    return device_service.monitor_series(station_id, metric, hours)


@router.post("/docs/upload", response_model=UploadResult, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    strategy: str = "fixed",
    kb_id: str = "kb_general",
) -> UploadResult:
    """上传文档 → 解析 → 分块 → 向量化入库。

    注意点：
    - 一次只收一个文件，多文件批量由客户端循环调用（简单、失败可定位）；
    - 解析失败（扫描件/空文本）会返回 422 且**不会写入任何脏数据**，
      这对应「宁可报错也不入库空块」的入库原则；
    - 同名文件再次上传 = 覆盖（按文件名 hash 定位旧 chunk 先删后写）。
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="缺少文件名")
    content = file.file.read()
    try:
        result = rag.index_document(file.filename, content, strategy=strategy, kb_id=kb_id)
    except ValueError as exc:
        # ValueError = 文档内容问题（扫描件、空文本、维度不符），不是服务器故障
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 —— 其它异常（如云端欠费）也翻译成人话
        raise HTTPException(status_code=502, detail=f"入库失败：{exc}") from exc
    trace_event("file", filename=result["document_id"], chunk_count=result["chunk_count"])
    return UploadResult(filename=file.filename, strategy=strategy, **result)
