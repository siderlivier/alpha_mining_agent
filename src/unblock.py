"""
解鎖：把「當初因為 DSL 缺運算子而失敗、如今運算子已實作」的假設放回佇列。

為什麼需要這支程式
------------------
Distill 每輪會把「假設可能成立、但現有 DSL 表達不出來」的訊號寫進
memory/dsl_limitations.jsonl；整理回合據此提出運算子提案，提案裡的
`evidence` 欄位正好就是「被這個缺口卡住的 attempt 編號」。

但運算子實作完成後，那些假設沒有任何機制會被重新提出——它們躺在
attempts 裡，而 Generate 只讀 learnings.md。等於每實作一個運算子，
就有一批已經付過錢、診斷過、確認「假設本身沒被證偽」的候選被浪費掉。

這支程式走 提案.evidence → attempts → 佇列 這條既有的鏈路，把它們撈回來。

用法
----
    python src/unblock.py                 # 只列出可解鎖的假設，不改檔
    python src/unblock.py --apply         # 寫進 learnings.md 的待驗證假設佇列
    python src/unblock.py --op streak_true --apply   # 只處理某個運算子
    python src/unblock.py --limit 5 --apply          # 一次最多放回幾條

已放回的會記在 memory/unblocked.json，不會重複放。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import Memory

# 只有這些失因代表「假設可能仍成立，只是表達不出來」——
# hypothesis_wrong 是機制被證偽、duplicate 是紅海，都不該解鎖。
UNBLOCKABLE = {"expression_bad", ""}


def implemented_ops(mem: Memory) -> list[dict]:
    p = mem.root / "operator_proposals.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return [o for o in d.get("implemented", []) if o.get("evidence")]


def load_state(mem: Memory) -> dict:
    p = mem.root / "unblocked.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"done": []}


def save_state(mem: Memory, st: dict):
    (mem.root / "unblocked.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def collect(mem: Memory, only_op: str | None = None) -> list[dict]:
    """回傳可解鎖的假設清單（已排除做過的、失因不符的）。"""
    done = set(load_state(mem)["done"])
    by_id = {r["id"]: r for r in mem.load_attempts()}
    out, seen = [], set()
    for op in implemented_ops(mem):
        if only_op and op["name"] != only_op:
            continue
        for aid in op["evidence"]:
            key = f"{op['name']}|{aid}"
            if key in done or key in seen or aid not in by_id:
                continue
            rec = by_id[aid]
            if (rec.get("failure_type") or "") not in UNBLOCKABLE:
                continue
            seen.add(key)
            out.append({"key": key, "op": op["name"], "attempt": aid,
                        "hypothesis": rec.get("hypothesis", ""),
                        "formula": rec.get("formula", ""),
                        "verdict": rec.get("verdict", ""),
                        "failure_type": rec.get("failure_type", ""),
                        "example": op.get("example", "")})
    return out


def main():
    ap = argparse.ArgumentParser(description="把因缺運算子而卡住的假設放回佇列")
    ap.add_argument("--apply", action="store_true", help="實際寫入（預設只預覽）")
    ap.add_argument("--op", help="只處理指定運算子")
    ap.add_argument("--limit", type=int, default=10, help="一次最多放回幾條")
    a = ap.parse_args()

    mem = Memory()
    mem.ensure()
    items = collect(mem, a.op)
    if not items:
        print("沒有可解鎖的假設。")
        print("（可能原因：運算子的 evidence 對應的 attempt 失因不是 expression_bad，"
              "或全部已解鎖過——見 memory/unblocked.json）")
        return

    print(f"可解鎖 {len(items)} 條（本次處理上限 {a.limit}）：\n")
    todo = items[:a.limit]
    for i, it in enumerate(todo, 1):
        print(f"{i}. [{it['op']}] {it['attempt']}（{it['verdict']}/{it['failure_type']}）")
        print(f"   假設：{it['hypothesis'][:90]}")
        print(f"   原公式：{it['formula']}")
        if it["example"]:
            print(f"   新運算子用法：{it['example']}")
        print()
    if len(items) > len(todo):
        print(f"（另有 {len(items) - len(todo)} 條未處理，調高 --limit 或下次再跑）\n")

    if not a.apply:
        print("預覽模式。確認無誤後加 --apply 寫入佇列。")
        return

    st = load_state(mem)
    added, full = 0, False
    for it in todo:
        line = (f"- 【解鎖】{it['hypothesis'][:110]}"
                f"（原受限於 DSL，{it['op']} 已實作；原公式 {it['formula']}，"
                f"證據 {it['attempt']}）")
        if mem.append_learning("待驗證假設佇列", line):
            st["done"].append(it["key"])
            added += 1
        else:
            full = True
            break
    st["last_run"] = datetime.now().isoformat(timespec="seconds")
    save_state(mem, st)
    print(f"已放回佇列 {added} 條，記錄於 memory/unblocked.json。")
    if full:
        print("⚠️ 佇列已達上限，其餘未放入——先跑 python src/consolidate.py 精簡佇列。")
    print("下一輪 Generate 就會看到它們（並可用 from_queue 回報消化）。")


if __name__ == "__main__":
    main()
