"""
M2：嘗試紀錄檢索工具（規格書 7.5——用標籤檢索取代向量庫，零 token 成本）。

供整理回合與人工審計調閱個案。預設輸出緊湊摘要（一行一筆），
--full 輸出完整 JSON。

執行範例：
  python src/query_attempts.py --category revenue_momentum
  python src/query_attempts.py --verdict rejected_stage2 --limit 10
  python src/query_attempts.py --grep 投信 --full
  python src/query_attempts.py --id A-0007 --full
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import Memory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category")
    ap.add_argument("--verdict")
    ap.add_argument("--grep", help="在 hypothesis/diagnosis/lesson/formula 全文搜尋")
    ap.add_argument("--id")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    recs = Memory().load_attempts()
    if a.id:
        recs = [r for r in recs if r.get("id") == a.id]
    if a.category:
        recs = [r for r in recs if r.get("category") == a.category]
    if a.verdict:
        recs = [r for r in recs if r.get("verdict") == a.verdict]
    if a.grep:
        key = a.grep.lower()
        recs = [r for r in recs if key in json.dumps(
            {k: r.get(k, "") for k in
             ("hypothesis", "diagnosis", "lesson", "formula")},
            ensure_ascii=False).lower()]

    recs = recs[-a.limit:]
    if a.full:
        print(json.dumps(recs, ensure_ascii=False, indent=1))
    else:
        for r in recs:
            print(f"{r['id']} [{r.get('verdict','?'):16s}] "
                  f"({r.get('category','?')}) {r.get('formula','')[:48]} "
                  f"| {r.get('lesson','')[:60]}")
    print(f"-- {len(recs)} 筆", file=sys.stderr)


if __name__ == "__main__":
    main()
