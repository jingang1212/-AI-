"""Windows 控制台中文/emoji 打印修复。

现象：Windows 控制台默认编码是 GBK，Python 脚本里 print 中文、✅ 这类字符会：
    UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'
或者打印出来是乱码（因为输出流按 GBK 编码，日志/管道按 UTF-8 解码）。

解决：把标准输出/错误流重新配置成 UTF-8。
这类「代码逻辑没错但一跑就乱码」的问题，是新手最容易卡半天的坑之一。
"""

from __future__ import annotations

import sys


def setup_console() -> None:
    """在脚本入口调用一次即可。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            # reconfigure 是 Python 3.7+ 的方法；重定向到文件时可能不支持，忽略即可
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
