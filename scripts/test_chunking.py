"""分块与父子拆分单测：不依赖网络。

验证：markdown-heading 的标题路径与字符区间、fixed 的精确区间、
父子拆分（parent_id / parent_text / 标题继承）。
用法：python scripts/test_chunking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag  # noqa: E402
from app.console import setup_console  # noqa: E402

setup_console()

MD = "# 一级标题\n\n## 二级标题\n\n这是第一段内容，包含一些说明。\n\n这是第二段内容，用来测试段落切分。"


def main() -> None:
    md_chunks = rag.chunk(MD, "markdown-heading")
    assert md_chunks, "markdown-heading 应产出至少一个 chunk"
    assert md_chunks[0]["heading_path"] == "一级标题 / 二级标题", md_chunks[0]["heading_path"]
    assert MD[md_chunks[0]["start"] : md_chunks[0]["end"]].splitlines()[0].strip() == "# 一级标题"
    print("✅ markdown-heading 标题路径与起始定位正确")

    fixed = rag.chunk(MD, "fixed", size=30, overlap=5)
    assert fixed and all(MD[c["start"] : c["end"]] == c["text"] for c in fixed), "fixed 字符区间必须精确"
    print("✅ fixed 分块字符区间精确")

    units = rag.split_parent_child(MD, "markdown-heading")
    assert units, "父子拆分应产出子块"
    assert all(u["parent_id"] and u["parent_text"] and u["heading_path"] for u in units)
    assert set(u["parent_id"] for u in units), "子块必须归属父块"
    print(f"✅ 父子拆分正确（{len(units)} 个子块，父块数 {len(set(u['parent_id'] for u in units))}）")
    print("全部通过")


if __name__ == "__main__":
    main()
