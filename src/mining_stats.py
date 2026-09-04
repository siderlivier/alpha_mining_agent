# -*- coding: utf-8 -*-
"""
挖礦迴圈的運作統計：漏斗淘汰組成、入庫率隨時間的變化、記憶與提案規模。

    python src/mining_stats.py

存在的理由：README 宣稱「入庫率下滑是因子庫飽和，不是 agent 退步」。
那是一個可被否證的主張，所以它需要一支能重跑的程式，而不是一段文字。
判準寫在最後的判讀裡：**看 S1／S4 與 S2 的走向是否分歧**。
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEM = ROOT / "memory"

STAGES = [
    ("rejected_syntax", "語法"),
    ("rejected_stage1", "S1 IC"),
    ("rejected_stage2", "S2 去相關"),
    ("rejected_stage3", "S3 批次"),
    ("rejected_stage4", "S4 品質"),
]


def load_attempts() -> list[tuple[int, str]]:
    out = []
    for f in glob.glob(str(MEM / "attempts" / "*.json")):
        try:
            a = json.load(io.open(f, encoding="utf-8"))
        except Exception:
            continue
        r, v = a.get("round"), a.get("verdict") or "?"
        if isinstance(r, int):
            out.append((r, v))
    return sorted(out)


def _counts(rows, lo, hi):
    c = collections.Counter(v for r, v in rows if lo <= r <= hi)
    return c, sum(c.values())


def main() -> None:
    rows = load_attempts()
    if not rows:
        raise SystemExit("memory/attempts/ 是空的")
    n = max(r for r, _ in rows)
    total_files = len(glob.glob(str(MEM / "attempts" / "*.json")))
    print(f"attempts 檔案 {total_files} 筆，其中有輪次標記 {len(rows)} 筆，"
          f"輪次 1 ~ {n}\n")

    hdr = f"{'區段':<20}{'候選':>6}" + "".join(f"{lab:>11}" for _, lab in STAGES) + f"{'入庫':>8}"
    print(hdr)
    print("-" * len(hdr.encode("utf-8")) // 2 * "-" if False else "-" * 84)

    segs = [(1, n // 3, "前 1/3"),
            (n // 3 + 1, 2 * n // 3, "中 1/3"),
            (2 * n // 3 + 1, n, "後 1/3")]
    trend = {}
    for lo, hi, lab in segs:
        c, t = _counts(rows, lo, hi)
        p = c["passed"] + c["passed_industry"]
        line = f"{lab + f' (輪{lo}-{hi})':<20}{t:>6}"
        for k, _ in STAGES:
            line += f"{c[k] / t * 100:>10.1f}%"
        line += f"{p / t * 100:>7.1f}%"
        print(line)
        trend[lab] = {k: c[k] / t * 100 for k, _ in STAGES} | {"入庫": p / t * 100}

    c, t = _counts(rows, 1, n)
    p = c["passed"] + c["passed_industry"]
    line = f"{'全期':<20}{t:>6}"
    for k, _ in STAGES:
        line += f"{c[k] / t * 100:>10.1f}%"
    line += f"{p / t * 100:>7.1f}%"
    print("-" * 84)
    print(line)
    print(f"\n入庫明細：一般 {c['passed']} 個 + 產業專屬 {c['passed_industry']} 個 = {p} 個")

    # --- 判讀：飽和 vs 退步 ---
    a, z = trend["前 1/3"], trend["後 1/3"]
    d_s1 = z["rejected_stage1"] - a["rejected_stage1"]
    d_s4 = z["rejected_stage4"] - a["rejected_stage4"]
    d_s2 = z["rejected_stage2"] - a["rejected_stage2"]
    print(f"\n=== 判讀 ===")
    print(f"S1（訊號強度）淘汰率變化 {d_s1:+.1f}pp　"
          f"S4（品質）{d_s4:+.1f}pp　S2（與既有因子重複）{d_s2:+.1f}pp")
    if d_s2 > 0 and d_s1 <= 0 and d_s4 <= 0:
        print("✅ 因子庫飽和：假設品質**變好**（S1／S4 淘汰率下降），"
              "但越來越常撞上既有因子（S2 上升）。")
        print("   瓶頸從「找得到訊號嗎」變成「找得到**新的**訊號嗎」。")
    elif d_s1 > 0 or d_s4 > 0:
        print("⚠️ 假設品質下降：S1 或 S4 的淘汰率上升，"
              "代表 agent 提的東西變差，不只是因子庫飽和。")
        print("   該檢查 learnings.md 是否累積了錯誤的經驗。")
    else:
        print("？走向不明確，樣本可能太少。")

    # --- 記憶與提案 ---
    print(f"\n=== 記憶與提案 ===")
    lm = io.open(MEM / "learnings.md", encoding="utf-8").read()
    print(f"learnings.md　　　　{len(lm.splitlines())} 行、{len(lm)} 字元（唯一注入 prompt 的記憶）")
    print(f"learnings_history　 {len(os.listdir(MEM / 'learnings_history'))} 份備份")
    op = json.load(io.open(MEM / "operator_proposals.json", encoding="utf-8"))
    impl = [x.get("name") for x in op.get("implemented", []) if isinstance(x, dict)]
    print(f"運算子提案　　　　　 上線 {len(op.get('implemented', []))}"
          f"（{', '.join(filter(None, impl))}）、"
          f"待審 {len(op.get('pending', []))}、否決 {len(op.get('rejected_log', []))}")
    for name, path in [("DSL 表達限制", "dsl_limitations.jsonl"),
                       ("佇列已消化", "queue_consumed.jsonl")]:
        f = MEM / path
        k = sum(1 for _ in io.open(f, encoding="utf-8")) if f.exists() else 0
        print(f"{name}　　　　 {k} 筆")

    b = json.load(io.open(MEM / "budget.json", encoding="utf-8"))
    tb = b.get("tokens_breakdown", {})
    out_share = tb.get("output", 0) / max(1, sum(tb.values())) * 100
    print(f"\n=== 成本（{b.get('week_of')} 那一週）===")
    print(f"累計輪次 {b.get('total_rounds')}　本週 {b.get('rounds_used')} 輪、"
          f"{b.get('tokens_used'):,} tokens、${b.get('cost_usd'):.2f}"
          f"（約 ${b.get('cost_usd') / max(1, b.get('rounds_used')):.2f}／輪）")
    print(f"輸出 token 佔 {out_share:.0f}%——成本主要花在推理，不是讀資料。")


if __name__ == "__main__":
    main()
