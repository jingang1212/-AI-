"""批量导入 data/docs 下的文档。

用法：
    python scripts/ingest.py                       # 默认 fixed 分块，增量导入（同名覆盖）
    python scripts/ingest.py markdown-heading      # 指定分块策略
    python scripts/ingest.py --reset               # 先清空向量库再全量导入
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, rag, vectorstore
from app.console import setup_console

setup_console()


def main():
    parser = argparse.ArgumentParser(description="把 data/docs 导入向量库")
    parser.add_argument(
        "strategy", nargs="?", default="fixed",
        choices=["fixed", "markdown-heading", "paragraph", "sentence"],
        help="分块策略（默认 fixed）",
    )
    parser.add_argument("--reset", action="store_true", help="先清空向量库再导入")
    args = parser.parse_args()

    if args.reset:
        vectorstore.reset()
        print("向量库已清空，开始全量导入")

    imported = []
    for p in sorted(config.DOCS_DIR.iterdir()):
        if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
            try:
                r = rag.index_document(p.name, p.read_bytes(), strategy=args.strategy)
                imported.append((p.name, r["chunk_count"]))
            except Exception as e:
                # EmbeddingError 里已经把「账户欠费/额度用尽」这类问题翻译成人话了
                print(f"跳过 {p.name}: {e}")
    for name, n in imported:
        print(f"已索引 {name}: {n} 块")
    print(f"共 {len(imported)} 篇")


if __name__ == "__main__":
    main()
