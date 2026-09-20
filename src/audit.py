"""
M4：聚合審計（規格書 9.2）。

研究回饋只計算入庫時同範圍的 sub-train → validation 衰減。
密封 test 不進入預設聚合或整理 prompt；--human 才能額外讀取 test。
重疊聚合不能保證個體無法反推，因此不得以聚合方式讓 test 回流挖礦。

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
from memory import Memory, admission_metrics

MIN_N = 2   # validation 聚合的最小樣本，不是 test 隔離保證


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


def collect(mem: Memory, *, human=False) -> list[dict]:
    """Validation only for research feedback; test requires explicit human mode."""
    rows = []
    for fid, m in mem.own_library().items():   # 參考因子不算 agent 的成果
        # Remove sealed data before deriving any default feedback statistics.
        safe = m if human else {k:v for k,v in m.items() if k!="test_metrics_sealed"}
        train, valid, sealed = admission_metrics(safe)
        tr = train.get("icir")
        te = (sealed if human else valid).get("icir")
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
        return (f"（聚合審計：目前僅 {len(rows)} 個因子有完整 train/validation 統計，"
                f"樣本不足 {MIN_N}，暫不輸出——避免反推個體數字）")
    lines = [f"[audit] 入庫時同範圍 train→validation ICIR 衰減聚合（n={len(rows)}，"
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
        rows = collect(mem, human=True)
        print("\n⚠️ 以下為人類專用明細（含個體 test 指標，勿餵 agent）：")
        for r in sorted(rows, key=lambda x: -x["decay"]):
            print(f"  {r['fid']} [{r['category']}/{r['aspect']}/d{r['depth']}] "
                  f"train {r['tr']:.2f} → test {r['te']:.2f}（衰減 {r['decay']:.0f}%）")


if __name__ == "__main__":
    main()
