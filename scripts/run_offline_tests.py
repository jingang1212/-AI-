"""离线测试入口：汇总不依赖网络的测试，任一失败即非零退出码。

用法：python scripts/run_offline_tests.py
CI 用本脚本做门禁。注意 test_milvus.py 会独占 Milvus Lite 文件锁，
运行前请确保没有正在运行的服务；需要模型/网络的测试（流式、检索评测）另行执行。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OFFLINE_TESTS = [
    "scripts/test_chunking.py",
    "scripts/test_confirm_unit.py",
    "scripts/test_milvus.py",
    "scripts/test_login_rate_limit.py",
]


def main() -> int:
    failed: list[str] = []
    for script in OFFLINE_TESTS:
        print(f"\n===== {script} =====")
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            failed.append(script)
    print(f"\n离线测试：通过 {len(OFFLINE_TESTS) - len(failed)}/{len(OFFLINE_TESTS)}")
    if failed:
        for name in failed:
            print(f"  失败：{name}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
