# -*- coding: utf-8 -*-
"""
橫斷面樣本外檢定：因子在「從來沒被挖礦看過的產業」上還有效嗎？

26 個自有因子（F-xxx）是在 4 個產業、958 檔股票上挖出來的。
資料修復後新增了 658 檔，其中**民生與製造（309 檔）與營建（89 檔）
是兩個完整的新產業——agent 在挖礦時一次都沒看過**。

這構成第二種樣本外，與時間序列的樣本外互相獨立：

    時間序列樣本外：往前走（test 期 2020-01 起）
    橫斷面樣本外：往旁邊走（新產業）  ← 這支在測的

判讀：若一個因子抓到的是真的經濟機制，它在新產業應該還有 IC；
若它只是原本 4 個產業的某種特性代理，在新產業就會歸零。

    python src/crosssec_oos.py                  # 全部 26 個
    python src/crosssec_oos.py --save out.json

⛔ 這支只讀資料、不寫任何 memory/ 檔案。因子庫裡的歷史指標是那次挖礦的
   紀錄，不該被這次的重算覆蓋掉。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsl
import eval_candidates as ec

ROOT = Path(__file__).resolve().parents[1]
MIN_STOCKS = ec.MIN_STOCKS

OLD_GROUPS = {"電子", "半導體", "生技", "金融"}          # 挖礦時看得到的
NEW_GROUPS = {"民生與製造", "營建"}                      # 挖礦時不存在的


def ic_series(ctx, fac: pd.DataFrame, groups: set, months) -> pd.Series:
    """限定在某幾個產業內算逐月 rank IC（口徑與 eval_candidates.monthly_ic 相同）。"""
    gcols = {g: c for g, c in ec.group_cols(ctx, fac).items() if g in groups}
    if not gcols:
        return pd.Series(dtype=float)
    recs = {}
    for ym in months:
        if ym not in fac.index:
            continue
        frow, rrow = fac.loc[ym], ctx.fwd.loc[ym]
        ics = []
        for g, cols in gcols.items():
            f = frow[cols].astype(float)
            r = rrow[cols].astype(float)
            ok = f.notna() & r.notna()
            if ok.sum() < MIN_STOCKS:
                continue
            fv, rv = f[ok], r[ok]
            if fv.nunique() < 2 or rv.nunique() < 2:
                continue
            ic = fv.rank().corr(rv.rank())
            if pd.notna(ic):
                ics.append(ic)
        if ics:
            recs[ym] = float(np.mean(ics))
    return pd.Series(recs).sort_index()


def stats(ic: pd.Series) -> dict:
    """IC 平均、ICIR、t 值、勝率。t = ICIR × √月數。"""
    ic = ic.dropna()
    if len(ic) < 12:
        return {"n": len(ic), "ic": np.nan, "icir": np.nan, "t": np.nan, "win": np.nan}
    sd = ic.std()
    icir = ic.mean() / sd if sd > 0 else np.nan
    return {"n": int(len(ic)), "ic": float(ic.mean()), "icir": float(icir),
            "t": float(icir * np.sqrt(len(ic))), "win": float((ic > 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", help="結果存成 JSON")
    ap.add_argument("--span", default="all",
                    choices=["all", "sub_train", "validation", "test"],
                    help="限定期間；預設 all（新產業的每一個月都是橫斷面樣本外）")
    a = ap.parse_args()

    ctx = ec.Context()
    lib = json.loads((ROOT / "memory" / "library.json").read_text(encoding="utf-8"))
    own = {k: v for k, v in lib.items() if k.startswith("F-")}

    present = set(ctx.group_map.values())
    print(f"面板產業：{sorted(present)}")
    miss = NEW_GROUPS - present
    if miss:
        raise SystemExit(f"❌ 新產業 {miss} 不在面板裡——monthly_base 還沒重建？")
    months = ctx.months if a.span == "all" else ctx.mask(a.span)
    print(f"期間：{a.span}（{months[0]} ~ {months[-1]}，{len(months)} 個月）")
    n_old = sum(1 for g in ctx.group_map.values() if g in OLD_GROUPS)
    n_new = sum(1 for g in ctx.group_map.values() if g in NEW_GROUPS)
    print(f"原 4 產業 {n_old} 檔（挖礦看過）　新 2 產業 {n_new} 檔（挖礦沒看過）\n")

    rows = []
    for fid, meta in sorted(own.items()):
        try:
            fac = dsl.compute(meta["formula"], ctx.data, ctx.group_map)
        except Exception as e:
            print(f"{fid} 計算失敗：{e}")
            continue
        o = stats(ic_series(ctx, fac, OLD_GROUPS, months))
        n = stats(ic_series(ctx, fac, NEW_GROUPS, months))
        sign = np.sign(o["ic"]) if pd.notna(o["ic"]) else np.nan
        # 方向一致才算「撐過去」：新產業的 IC 要同號，且 |t| ≥ 2
        held = (pd.notna(n["t"]) and pd.notna(sign)
                and np.sign(n["ic"]) == sign and abs(n["t"]) >= 2.0)
        rows.append({"factor": fid, "name": meta.get("name_zh", ""),
                     "scope": meta.get("industry_scope"), "old": o, "new": n,
                     "held": bool(held)})

    rows.sort(key=lambda r: -(abs(r["new"]["t"]) if pd.notna(r["new"]["t"]) else -1))
    print(f"{'因子':<7}{'名稱':<16}{'原 4 產業':>26}{'新 2 產業':>26}   判定")
    print(f"{'':<7}{'':<16}{'IC':>8}{'ICIR':>7}{'t':>7}"
          f"{'IC':>10}{'ICIR':>7}{'t':>7}")
    print("-" * 104)
    f3 = lambda x: f"{x:>8.4f}" if pd.notna(x) else f"{'—':>8}"
    f2 = lambda x, w=7: f"{x:>{w}.2f}" if pd.notna(x) else f"{'—':>{w}}"
    for r in rows:
        o, n = r["old"], r["new"]
        tag = "✅ 撐過去" if r["held"] else ("⚠️ 只在原產業" if pd.notna(n["t"]) else "— 樣本不足")
        star = "*" if r["scope"] else " "
        print(f"{r['factor']:<7}{r['name'][:15]:<16}{f3(o['ic'])}{f2(o['icir'])}{f2(o['t'])}"
              f"{f3(n['ic']):>10}{f2(n['icir'])}{f2(n['t'])}   {tag}{star}")

    held = [r for r in rows if r["held"]]
    valid = [r for r in rows if pd.notna(r["new"]["t"])]
    print("-" * 104)
    print(f"\n{len(held)}/{len(valid)} 個因子在**從沒看過的產業**上維持同向且 |t| ≥ 2")
    print("（* = 該因子入庫時被限定在某個產業，本表仍在全部產業上測，供對照）")
    print("\n⚠️ 判讀限制：新產業的樣本是同一段歷史，所以這是**橫斷面**的樣本外，"
          "\n   不是時間序列的。兩者互相獨立，都通過才比較站得住。")

    if a.save:
        Path(a.save).write_text(json.dumps(rows, ensure_ascii=False, indent=1,
                                           default=float), encoding="utf-8")
        print(f"\n已存 {a.save}")


if __name__ == "__main__":
    main()
