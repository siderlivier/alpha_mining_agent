"""
M4：聚合審計（規格書 9.2）。

對入庫因子計算 sub-train → test 的 ICIR 衰減，但輸出**只按維度聚合**：
category / AST 深度 / 運算子 / 欄位面向 / 窗口參數。
洩漏控制：絕不輸出任何單一因子的 test 數字——agent 只能學到「哪類設計
容易過擬合」的原則，無法藉此挑選特定因子。

audit_text() 的輸出會作為整理回合的輸入；--human 模式額外含樣本明細供人檢查。

執行：python src/audit.py            # 聚合版（可餵 agent）
     python src/audit.py --human    # 檢查版（含因子清單，勿餵 agent）
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsl
from memory import Memory

MIN_N = 2   # 聚合組至少幾個因子才輸出（單一因子的組會反推出個體數字）


def _ops_of(formula: str) -> set:
    try:
        pf = dsl.parse(formula)
    except dsl.DSLError:
        return set()
    ops = set()

    def walk(n):
        if n[0] == "call":
            ops.add(n[1])
            for a in n[2]:
                walk(a)
    walk(pf.tree)
    return ops


def _windows_of(formula: str) -> set:
    import re
    return set(int(x) for x in re.findall(r",\s*(\d+)\s*\)", formula))


def collect(mem: Memory) -> list[dict]:
    """每個入庫因子一筆：維度標籤 + train/test icir（內部用，不外流）。"""
    rows = []
    for fid, m in mem.own_library().items():   # 參考因子不算 agent 的成果
        sealed = m.get("test_metrics_sealed") or {}
        tr = (m.get("sub_train") or {}).get("icir")
        te = sealed.get("icir")
        if tr is None or te is None or tr <= 0:
            continue
        rows.append({
            "fid": fid, "category": m.get("category", "?"),
            "aspect": m.get("aspect", "?"), "depth": m.get("depth"),
            "ops": _ops_of(m.get("formula", "")),
            "windows": _windows_of(m.get("formula", "")),
            "tr": tr, "te": te, "decay": (1 - te / tr) * 100,
        })
    return rows


def _agg(rows, key_fn) -> dict:
    groups = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if isinstance(k, (set, frozenset)):
            for kk in k:
                groups[kk].append(r["decay"])
        elif k is not None:
            groups[k].append(r["decay"])
    return {k: (len(v), float(np.mean(v))) for k, v in groups.items()
            if len(v) >= MIN_N}


def audit_text(mem: Memory) -> str:
    """聚合審計文字（可安全餵給整理回合）。"""
    rows = collect(mem)
    if len(rows) < MIN_N:
        return (f"（聚合審計：目前僅 {len(rows)} 個因子有完整 train/test 統計，"
                f"樣本不足 {MIN_N}，暫不輸出——避免反推個體數字）")
    lines = [f"[audit] 入庫因子 train→test ICIR 衰減聚合（n={len(rows)}，"
             f"全體平均衰減 {np.mean([r['decay'] for r in rows]):.0f}%）："]
    dims = [("category", lambda r: r["category"]),
            ("面向", lambda r: r["aspect"]),
            ("AST深度", lambda r: f"深度{r['depth']}"),
            ("運算子", lambda r: r["ops"]),
            ("窗口", lambda r: {f"n={w}" for w in r["windows"]})]
    for name, fn in dims:
        agg = _agg(rows, fn)
        if not agg:
            continue
        parts = [f"{k}: 衰減{v[1]:.0f}%(n={v[0]})"
                 for k, v in sorted(agg.items(), key=lambda x: -x[1][1])]
        lines.append(f"  {name} → " + "；".join(parts))
    lines.append("  （衰減越高越可能過擬合；此為聚合統計，禁止據此推論單一因子）")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", action="store_true")
    a = ap.parse_args()
    mem = Memory()
    print(audit_text(mem))
    if a.human:
        rows = collect(mem)
        print("\n⚠️ 以下為人類專用明細（含個體 test 指標，勿餵 agent）：")
        for r in sorted(rows, key=lambda x: -x["decay"]):
            print(f"  {r['fid']} [{r['category']}/{r['aspect']}/d{r['depth']}] "
                  f"train {r['tr']:.2f} → test {r['te']:.2f}（衰減 {r['decay']:.0f}%）")


if __name__ == "__main__":
    main()
