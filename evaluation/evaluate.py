"""检索评测：recall@k。用法：python evaluation/evaluate.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import rag


def evaluate(eval_file: str, top_k: int) -> float:
    cases = json.loads(Path(eval_file).read_text(encoding="utf-8"))
    hits = 0
    for c in cases:
        sources = [h["source"] for h in rag.search(c["query"], top_k=top_k)]
        if c["expected"] in sources:
            hits += 1
    recall = hits / len(cases) if cases else 0.0
    print(f"top_k={top_k}  命中 {hits}/{len(cases)}  recall@{top_k}={recall:.2%}")
    return recall


if __name__ == "__main__":
    file = sys.argv[1] if len(sys.argv) > 1 else "evaluation/eval_set.json"
    for k in (3, 5, 8):
        evaluate(file, k)
