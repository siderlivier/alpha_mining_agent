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


# ---------------------------------------------------------------------------
# 逐因子對照：LLM 挖出來的公式 vs 原始欄位基準
# ---------------------------------------------------------------------------

def _approved(meta: dict) -> set:
    """因子的核准產業。參考因子與全池因子都有 approved_groups。"""
    ag = meta.get("approved_groups")
    if isinstance(ag, str):          # library.json 有些欄位存成字串化的 list
        try:
            ag = json.loads(ag.replace("'", '"'))
        except Exception:
            ag = None
    if isinstance(ag, list) and ag:
        return set(ag)
    sc = meta.get("industry_scope")
    return {sc} if sc else set()


def _values_matrix(fv: pd.DataFrame, fid: str, months, cols) -> pd.DataFrame:
    """從 factor_values.parquet 取出寬表，對齊 Context 的月份與股票欄。"""
    g = fv[fv["factor_id"] == fid]
    if not len(g):
        return pd.DataFrame()
    return (g.pivot_table(index="ym", columns="stock_id", values="value",
                          aggfunc="first")
             .reindex(months).reindex(columns=cols))


def cmd_vs_reference(save: str | None = None) -> list:
    """
    把 26 個自有因子與 18 個參考因子放在同一張表上逐一比較。

    為什麼是「逐因子」而不是「合成組合」
    ------------------------------------
    回測是**逐月在各產業內**取前 10%，等於每個產業各自等權。但自有因子裡
    有不少是產業限定的（approved_groups 只有 1~2 個產業），把它們和全池因子
    合成同一個分數，等於讓一個只在電子有效的因子去替營建的股票打分。
    合成績效因此混雜了「因子有沒有用」與「產業配置對不對」兩件事。

    逐因子評估沒有這個問題：**每個因子只在自己的核准產業裡量 IC**。

    為什麼這個比較是公平的
    ----------------------
    兩邊都只用 sub_train + validation 選出來（參考因子的
    `selection_periods` 明確記錄了這件事），所以 **test 期對兩邊
    同樣是樣本外**，比較是對稱的。

    ⚠️ 輸出含 test 期指標，屬人類研究報告，**不得進入任何 prompt**。
    """
    ctx = ec.Context()
    lib = json.loads((ROOT / "memory" / "library.json").read_text(encoding="utf-8"))
    fv = pd.read_parquet(ROOT / "memory" / "factor_values.parquet")
    cols = ctx.fwd.columns

    spans = ["sub_train", "validation", "test"]
    rows = []
    for fid, meta in sorted(lib.items()):
        groups = _approved(meta)
        if not groups:
            continue
        fac = _values_matrix(fv, fid, ctx.months, cols)
        if not len(fac):
            continue
        rec = {"factor": fid, "name": meta.get("name_zh", ""),
               "kind": "reference" if fid.startswith("R-") else "own",
               "n_groups": len(groups), "groups": sorted(groups)}
        for sp in spans:
            rec[sp] = stats(ic_series(ctx, fac, groups, ctx.mask(sp)))
        rows.append(rec)

    own = [r for r in rows if r["kind"] == "own"]
    ref = [r for r in rows if r["kind"] == "reference"]

    def _tbl(items, title):
        print(f"\n===== {title}（{len(items)} 個）=====")
        print(f"{'id':<7}{'名稱':<22}{'產業':>4}"
              f"{'train ICIR':>12}{'valid ICIR':>12}{'⚠test ICIR':>12}{'⚠test t':>10}")
        print("-" * 82)
        f2 = lambda x: f"{x:>12.2f}" if pd.notna(x) else f"{'—':>12}"
        for r in sorted(items, key=lambda r: -(r["test"]["icir"]
                                               if pd.notna(r["test"]["icir"]) else -9)):
            print(f"{r['factor']:<7}{r['name'][:21]:<22}{r['n_groups']:>4}"
                  f"{f2(r['sub_train']['icir'])}{f2(r['validation']['icir'])}"
                  f"{f2(r['test']['icir'])}"
                  f"{r['test']['t']:>10.2f}" if pd.notna(r["test"]["t"])
                  else f"{r['factor']:<7}{r['name'][:21]:<22}{r['n_groups']:>4}"
                       f"{f2(r['sub_train']['icir'])}{f2(r['validation']['icir'])}"
                       f"{f2(r['test']['icir'])}{'—':>10}")

    _tbl(own, "自有因子（LLM 提假設 → DSL 公式）")
    _tbl(ref, "參考因子（原始欄位，方向修正後）")

    print(f"\n===== 分布對照 =====")
    print(f"{'':<14}{'n':>4}{'test ICIR 中位':>14}{'平均':>8}"
          f"{'ICIR>0.3':>10}{'|t|≥2':>8}{'中位覆蓋產業數':>14}")
    print("-" * 74)
    for items, lab in [(own, "自有"), (ref, "參考")]:
        ic = [r["test"]["icir"] for r in items if pd.notna(r["test"]["icir"])]
        tt = [r["test"]["t"] for r in items if pd.notna(r["test"]["t"])]
        ng = [r["n_groups"] for r in items]
        print(f"{lab:<14}{len(items):>4}{np.median(ic):>14.3f}{np.mean(ic):>8.3f}"
              f"{sum(1 for x in ic if x > 0.3):>10}{sum(1 for x in tt if abs(x) >= 2):>8}"
              f"{int(np.median(ng)):>14}")

    print("\n⚠️ 兩邊都只用 sub_train + validation 選出，所以 test 期對兩邊同樣是樣本外，"
          "\n   比較是對稱的。但 test 期已被歷次研究使用過，不是全新 holdout。")
    print("⚠️ 每個因子只在自己的 approved_groups 內量 IC——產業限定因子不會被"
          "\n   拿去替它沒核准的產業打分。這正是不做合成績效的理由。")

    if save:
        Path(save).write_text(json.dumps(rows, ensure_ascii=False, indent=1,
                                         default=float), encoding="utf-8")
        print(f"\n已存 {save}")
    return rows


# ---------------------------------------------------------------------------
# 逐對比較：每個自有因子 × 每個基準因子 × 該自有因子的每個核准產業
# ---------------------------------------------------------------------------

def _median_stocks(ctx, fac: pd.DataFrame, group: str, months) -> float:
    """該產業每月「可評分股票數」的中位數——讀 max/min 與全距時必須配著看。"""
    sids = [s for s, g in ctx.group_map.items() if g == group and s in fac.columns]
    ns = []
    for ym in months:
        if ym not in fac.index:
            continue
        f = fac.loc[ym][sids].astype(float)
        r = ctx.fwd.loc[ym][sids].astype(float)
        ns.append(int((f.notna() & r.notna()).sum()))
    return float(np.median(ns)) if ns else float("nan")


def _ic_one_group(ctx, fac: pd.DataFrame, group: str, months) -> pd.Series:
    """單一產業內的逐月 rank IC（口徑與 eval_candidates.monthly_ic 相同）。"""
    sids = [s for s, g in ctx.group_map.items() if g == group and s in fac.columns]
    rec = {}
    for ym in months:
        if ym not in fac.index:
            continue
        f = fac.loc[ym][sids].astype(float)
        r = ctx.fwd.loc[ym][sids].astype(float)
        ok = f.notna() & r.notna()
        if ok.sum() < MIN_STOCKS:
            continue
        fv_, rv_ = f[ok], r[ok]
        if fv_.nunique() < 2 or rv_.nunique() < 2:
            continue
        v = fv_.rank().corr(rv_.rank())
        if pd.notna(v):
            rec[ym] = float(v)
    return pd.Series(rec, dtype=float).sort_index()


def _desc(x: pd.Series) -> dict:
    """單一 IC 序列的四個描述量。全距 = max − min。"""
    sd = x.std()
    return {"mean": float(x.mean()), "max": float(x.max()), "min": float(x.min()),
            "range": float(x.max() - x.min()),
            "icir": float(x.mean() / sd) if sd > 0 else np.nan}


def cmd_pairwise(save_csv: str, spans=("sub_train", "validation", "test"),
                 min_months: int = 12) -> pd.DataFrame:
    """
    每個自有因子對上每個基準因子，**只在該自有因子的核准產業內**比較。

    為什麼限定在核准產業
    --------------------
    自有因子裡有不少是產業限定的（`approved_groups` 只有 1~2 個產業）。
    拿它去跟一個全池基準因子在「它沒核准的產業」比，比的是它本來就不宣稱
    有效的地方，對自有因子不公平；反過來只看它的主場，對基準因子也要在
    同一個主場量，才是同一條起跑線。所以**兩邊都只在該產業內算**。

    為什麼用全距而不是 max／min 各自相減
    ------------------------------------
    `max(IC_own) − max(IC_ref)` 的兩個極值可能落在**不同月份**，相減沒有
    物理意義。全距（max − min）描述的是同一條序列自己的離散程度，兩邊各自
    算完再比，才解釋得通。

    ⚠️ 符號慣例（唯一一個「正值代表較差」的欄位）
    ----------------------------------------------
        d_ic_mean   > 0  自有因子訊號較強      ✅
        d_ic_range  > 0  自有因子**波動較大**  ⚠️ 較差
        d_icir      > 0  自有因子穩定度較好    ✅

    ⚠️ 本表含 test 期指標，屬人類研究報告，**不得進入任何 prompt**。
    """
    ctx = ec.Context()
    lib = json.loads((ROOT / "memory" / "library.json").read_text(encoding="utf-8"))
    fv = pd.read_parquet(ROOT / "memory" / "factor_values.parquet")
    cols = ctx.fwd.columns

    own = {k: v for k, v in sorted(lib.items()) if k.startswith("F-")}
    ref = {k: v for k, v in sorted(lib.items()) if k.startswith("R-")}
    print(f"自有因子 {len(own)} 個、基準因子 {len(ref)} 個、產業 "
          f"{len(set(ctx.group_map.values()))} 個")

    mats = {fid: _values_matrix(fv, fid, ctx.months, cols)
            for fid in list(own) + list(ref)}
    need_groups = sorted({g for m in own.values() for g in _approved(m)})

    # 先把 (因子, 產業) 的 IC 序列算一次，之後各 span 只做切片——
    # 不這樣做的話 1500+ 組配對會把同一條序列重算幾十次。
    print(f"預先計算 IC 序列（{len(mats)} 因子 × {len(need_groups)} 產業）…")
    ic_cache, ns_cache = {}, {}
    for fid, fac in mats.items():
        if not len(fac):
            continue
        for g in need_groups:
            ic_cache[(fid, g)] = _ic_one_group(ctx, fac, g, ctx.months)
            ns_cache[(fid, g)] = _median_stocks(ctx, fac, g, ctx.months)

    rows = []
    for oid, ometa in own.items():
        for g in sorted(_approved(ometa)):
            a_all = ic_cache.get((oid, g))
            if a_all is None or not len(a_all):
                continue
            for rid, rmeta in ref.items():
                b_all = ic_cache.get((rid, g))
                if b_all is None or not len(b_all):
                    continue
                for sp in spans:
                    ms = set(ctx.mask(sp))
                    idx = [m for m in a_all.index if m in ms and m in b_all.index]
                    if len(idx) < min_months:
                        continue
                    A, B = _desc(a_all[idx]), _desc(b_all[idx])
                    rows.append({
                        "own_id": oid, "own_name": ometa.get("name_zh", ""),
                        "ref_id": rid, "ref_name": rmeta.get("name_zh", ""),
                        "group": g, "span": sp, "n_months": len(idx),
                        "median_stocks": ns_cache.get((oid, g)),
                        "own_ic_mean": A["mean"], "ref_ic_mean": B["mean"],
                        "d_ic_mean": A["mean"] - B["mean"],
                        "own_ic_max": A["max"], "own_ic_min": A["min"],
                        "ref_ic_max": B["max"], "ref_ic_min": B["min"],
                        "own_ic_range": A["range"], "ref_ic_range": B["range"],
                        "d_ic_range": A["range"] - B["range"],
                        "own_icir": A["icir"], "ref_icir": B["icir"],
                        "d_icir": A["icir"] - B["icir"],
                    })

    df = pd.DataFrame(rows)
    Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(save_csv, index=False, encoding="utf-8-sig")
    print(f"\n已存 {save_csv}　{len(df):,} 列 × {len(df.columns)} 欄")

    t = df[df["span"] == "test"]
    if len(t):
        print(f"\n===== test 期摘要（{len(t):,} 組配對）=====")
        print(f"{'':<16}{'勝過基準的比例':>16}{'中位差':>10}")
        for c, lab, good in [("d_ic_mean", "平均 IC 較高", True),
                             ("d_icir", "ICIR 較高", True),
                             ("d_ic_range", "全距較小（較穩）", False)]:
            v = t[c].dropna()
            win = (v > 0).mean() if good else (v < 0).mean()
            print(f"{lab:<16}{win * 100:>15.1f}%{v.median():>10.4f}")

        print(f"\n逐因子（test 期，對 18 個基準因子跨核准產業平均）：")
        gp = (t.groupby(["own_id", "own_name"])
                .agg(產業數=("group", "nunique"), 配對數=("ref_id", "size"),
                     平均IC差=("d_ic_mean", "mean"), ICIR差=("d_icir", "mean"),
                     全距差=("d_ic_range", "mean"))
                .sort_values("ICIR差", ascending=False))
        print(gp.round(4).to_string())
    print("\n⚠️ d_ic_range 是唯一「正值代表較差」的欄位（自有因子波動較大）。")
    print("⚠️ median_stocks 小的產業（例如金融約 45 檔），單月 IC 的極值主要"
          "\n   反映樣本雜訊，全距會被放大——讀全距一定要配著這一欄看。")
    return df


def main() -> None:
    scope_flags = {"--refresh", "--check-only", "--auto-apply", "--apply", "--recover", "--requalify"}
    if scope_flags.intersection(sys.argv[1:]):
        import factor_scope
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg not in ("--refresh", "--requalify")]
        raise SystemExit(factor_scope.main())
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise", action="store_true",
                    help="逐對比較並輸出 CSV：自有 × 基準 × 核准產業")
    ap.add_argument("--csv", default="logs/pairwise_own_vs_reference.csv",
                    help="--pairwise 的輸出路徑")
    ap.add_argument("--vs-reference", action="store_true",
                    help="逐因子對照：自有公式 vs 原始欄位基準（只在各自核准產業內量）")
    ap.add_argument("--save", help="結果存成 JSON")
    ap.add_argument("--span", default="all",
                    choices=["all", "sub_train", "validation", "test"],
                    help="限定期間；預設 all（新產業的每一個月都是橫斷面樣本外）")
    a = ap.parse_args()

    if a.pairwise:
        cmd_pairwise(a.csv)
        return

    if a.vs_reference:
        cmd_vs_reference(a.save)
        return

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
            fac = ec.orient_factor(fac, meta)
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
