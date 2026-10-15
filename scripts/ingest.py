"""批量导入 data/docs 下的文档。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, rag


def main():
    imported = []
    strategy = sys.argv[1] if len(sys.argv) > 1 else "fixed"
    for p in sorted(config.DOCS_DIR.iterdir()):
        if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}:
            try:
                r = rag.index_document(p.name, p.read_bytes(), strategy=strategy)
                imported.append((p.name, r["chunk_count"]))
            except Exception as e:
                print(f"跳过 {p.name}: {e}")
    for name, n in imported:
        print(f"已索引 {name}: {n} 块")
    print(f"共 {len(imported)} 篇")


if __name__ == "__main__":
    main()
