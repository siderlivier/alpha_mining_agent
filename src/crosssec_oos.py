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


# ---------------------------------------------------------------------------
# 去膨脹 ICIR：直接對 IC 序列做多重檢定校正，不經過回測引擎
# ---------------------------------------------------------------------------

def _trial_icirs():
    """
    從 `memory/attempts/` 取出**實際搜尋歷史**的 sub_train ICIR。

    這是整支程式最關鍵的一塊。多數研究做不到完整的多重檢定校正，因為他們
    不留被拒候選的紀錄——只剩存活者，試驗數就只能用存活數，嚴重低估搜尋量。
    本專案的 `attempts/` 是**只進不出**，連死在 Stage 1 的都留著指標，
    所以試驗池可以是真正的搜尋歷史。
    """
    import ast
    rows = []
    for f in (ROOT / "memory" / "attempts").glob("*.json"):
        try:
            a = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        d = a.get("result")
        if isinstance(d, str):
            try:
                d = ast.literal_eval(d)
            except Exception:
                d = None
        if not isinstance(d, dict):
            continue
        st = d.get("sub_train") or {}
        if st.get("icir") is None:
            continue
        rows.append((a.get("verdict", "?"), float(st["icir"])))
    return np.array([r[1] for r in rows]), rows


def _expected_max(sigma: float, n: int) -> float:
    """N 次獨立抽樣中最大值的期望（Bailey & López de Prado 的近似式）。"""
    from scipy.stats import norm
    g = 0.5772156649                                   # Euler–Mascheroni
    return float(sigma * ((1 - g) * norm.ppf(1 - 1.0 / n)
                          + g * norm.ppf(1 - 1.0 / (n * np.e))))


def cmd_deflate_icir(span: str = "sub_train", save: str | None = None) -> list:
    """
    問題：這些因子是從 1,011 次搜尋裡挑出來的，它們的 ICIR 會不會只是運氣？

    為什麼不用既有的 `ml_diagnose --dsr`
    ------------------------------------
    `--dsr` 為了拿一個多重檢定校正，繞道走完了整個回測引擎：
    因子 → 產業內前 10% 組合 → 扣換手成本 → 主動報酬 → Sharpe → 去膨脹。
    於是它同時扛上 R05（缺報酬篩選）與 R12（換手定義）——而那兩件事跟
    「這個 ICIR 是不是挑出來的」完全無關。

    **ICIR 在數學形式上就是 Sharpe**：

        Sharpe = mean(月報酬) / std(月報酬)
        ICIR   = mean(月 IC)  / std(月 IC)      ← 同一個估計量

    去膨脹公式吃的是「一條序列 + N 個試驗」，不在乎序列是報酬還是 IC，
    所以直接對 IC 序列做，完全不碰回測引擎。

    ⚠️ 虛無分布要用哪一個——這是最容易算錯的地方
    ----------------------------------------------
    門檻 =「N 次試驗中最大值的期望」= σ × f(N)，關鍵在 σ 取什麼：

      (A) 1,011 個候選 ICIR 的**經驗**標準差（≈0.287）
          Bailey & López de Prado 原文的估計方式。但它混入了**候選之間
          真實的強弱差別**，不只是抽樣雜訊。

      (B) 無技巧的**理論**標準差 1/√T（T=72 時 ≈0.118）
          一個沒有預測力的因子，月 IC 均值為 0、ICIR 的抽樣誤差就是 1/√T。
          這才是「純運氣」該有的分布。

    (A) 約為 (B) 的 2.4 倍——那個差距是真實訊號差異，不是運氣。用 (A) 當
    虛無分布，等於假設「所有候選都沒有訊號、觀察到的離散全是運氣」，
    會把門檻灌到天上去。

    **本函式以 (B) 為主答案，同時列出 (A) 當保守上界。**

    ⚠️ test 期為什麼不需要做這個校正
    ---------------------------------
    選取只用了 sub_train + validation，**test 從未參與挑選**——沒有選擇偏誤
    就不需要校正，test 期的 ICIR 直接報即可。
    （但 test 期已被歷次研究反覆檢視，不是全新 holdout。那是另一種汙染，
      去膨脹修不了，只能靠「往前走」解決。）
    """
    import ml_diagnose as md                       # 借用 psr，不重複實作
    ctx = ec.Context()
    lib = json.loads((ROOT / "memory" / "library.json").read_text(encoding="utf-8"))
    fv = pd.read_parquet(ROOT / "memory" / "factor_values.parquet")
    cols = ctx.fwd.columns

    trials, rows = _trial_icirs()
    n_rej = sum(1 for v, _ in rows if not v.startswith("passed"))
    print(f"試驗池：**實際搜尋歷史** {len(trials)} 個候選"
          f"（含 {n_rej} 個被漏斗刷掉的）")
    print(f"  經驗 σ = {trials.std(ddof=1):.4f}　中位 {np.median(trials):.3f}"
          f"　max {trials.max():.3f}")

    own = {k: v for k, v in sorted(lib.items()) if k.startswith("F-")}
    months = ctx.mask(span)
    out = []
    for fid, meta in own.items():
        groups = _approved(meta)
        fac = _values_matrix(fv, fid, ctx.months, cols)
        if not len(fac) or not groups:
            continue
        ic = ic_series(ctx, fac, groups, months).dropna()
        if len(ic) < 24 or not (ic.std() > 0):
            continue
        out.append({"factor": fid, "name": meta.get("name_zh", ""),
                    "icir": float(ic.mean() / ic.std()), "T": int(len(ic)),
                    "skew": float(ic.skew()),
                    "kurt": float(ic.kurtosis() + 3.0)})
    if not out:
        raise SystemExit("沒有可評估的因子")

    T = int(np.median([o["T"] for o in out]))
    N = len(trials)
    sr0_null = _expected_max(1.0 / np.sqrt(T), N)         # (B) 主答案
    sr0_emp = _expected_max(float(trials.std(ddof=1)), N)  # (A) 保守上界

    print(f"\n虛無分布與門檻（N = {N}，T = {T} 個月）")
    print(f"  (B) 無技巧 σ = 1/√{T} = {1 / np.sqrt(T):.4f}"
          f"　→ 純運氣期望最大 ICIR = **{sr0_null:.3f}**  ← 主答案")
    print(f"  (A) 經驗 σ = {trials.std(ddof=1):.4f}"
          f"　→ 門檻 = {sr0_emp:.3f}  ← 保守上界（混入真實強弱差別，偏高）")

    for o in out:
        o["deflated_p"] = md.psr(o["icir"], sr0_null, o["T"], o["skew"], o["kurt"])
        o["pass_null"] = bool(o["icir"] > sr0_null)
        o["pass_emp"] = bool(o["icir"] > sr0_emp)

    out.sort(key=lambda x: -x["icir"])
    print(f"\n===== 逐因子（{span} 期，各自的核准產業內）=====")
    print(f"{'id':<7}{'名稱':<22}{'ICIR':>7}{'T':>5}{'去膨脹機率':>13}"
          f"{'過(B)':>7}{'過(A)':>7}")
    print("-" * 70)
    for o in out:
        print(f"{o['factor']:<7}{o['name'][:21]:<22}{o['icir']:>7.3f}{o['T']:>5}"
              f"{o['deflated_p']:>13.3f}"
              f"{'✅' if o['pass_null'] else '❌':>7}"
              f"{'✅' if o['pass_emp'] else '❌':>7}")

    nb = sum(o["pass_null"] for o in out)
    na = sum(o["pass_emp"] for o in out)
    print("-" * 70)
    print(f"以 (B) 主答案：**{nb} / {len(out)}** 個超過純運氣門檻 {sr0_null:.3f}")
    print(f"以 (A) 保守上界：{na} / {len(out)} 個超過 {sr0_emp:.3f}")
    print(f"去膨脹機率 > 0.95 的：{sum(o['deflated_p'] > 0.95 for o in out)} / {len(out)}")

    print(f"\n⚠️ 這個校正回答的是「**選取階段**有沒有被運氣主導」，"
          f"所以用 {span} 期——\n   因子實際就是在這裡被挑出來的。")
    print("⚠️ test 期不需要做這個校正：選取從未使用 test，沒有選擇偏誤。"
          "\n   但 test 期已被歷次研究反覆檢視，不是全新 holdout。")
    print("⚠️ 1,011 個候選彼此高度相關（Stage 2 存在的理由就是殺重複），"
          "\n   獨立性假設不成立 → 有效試驗數小於 1,011 → 真實門檻比 (B) 更低。"
          "\n   也就是說 (B) 已經站在偏保守的一側。")

    if save:
        Path(save).write_text(json.dumps(
            {"span": span, "n_trials": N, "T": T,
             "sr0_null": sr0_null, "sr0_empirical": sr0_emp, "factors": out},
            ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"\n已存 {save}")
    return out


def main() -> None:
    scope_flags = {"--refresh", "--check-only", "--auto-apply", "--apply", "--recover", "--requalify"}
    if scope_flags.intersection(sys.argv[1:]):
        import factor_scope
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg not in ("--refresh", "--requalify")]
        raise SystemExit(factor_scope.main())
    ap = argparse.ArgumentParser()
    ap.add_argument("--deflate-icir", action="store_true",
                    help="多重檢定校正：用完整搜尋歷史當試驗池，直接對 IC 序列去膨脹")
    ap.add_argument("--deflate-span", default="sub_train",
                    choices=["sub_train", "validation"],
                    help="校正的期間（選取實際發生的地方；test 不需要校正）")
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

    if a.deflate_icir:
        cmd_deflate_icir(a.deflate_span, a.save)
        return

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
