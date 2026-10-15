"""检索测试：python scripts/query.py "二氧化硫排放限值" 5"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "二氧化硫排放限值"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    hits = rag.search(query, top_k=top_k)
    print(f"问题：{query}")
    if not hits:
        print("无命中")
        return
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h['source']}] score={h['score']}")
        print(f"   {h['text'][:120]}")


if __name__ == "__main__":
    main()
