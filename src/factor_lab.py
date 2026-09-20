"""
因子組合實驗室：合成、比較、留一法診斷、貪婪前向選擇。

回答三個問題：
  1. agent 挖的因子 vs 前置專案 DFS 的參考因子，哪一組合成出來的組合比較好？
  2. 哪些因子在組合裡實際上是**拖累**（拿掉反而變好）？
  3. 從空集開始貪婪加入，最佳組合長什麼樣？

資料來源全部在本專案內，不跨專案：
  memory/factor_values.parquet   因子值（F-xxx 自有、R-xxx 前置專案匯入）
  data/monthly_base.parquet      fwd_ret_1m 標籤與產業別

⛔ 前瞻紀律（三道）
  1. **切分**：因子選取期的回測是 in-sample；歷史 DFS 池曾參考 test，
     不能因使用相同 test 起點就宣稱全新 holdout。跨專案比較另見 compare_upstream。
  2. **walk-forward + embargo**：模型只用 `ym <= months[i-1-EMBARGO]` 訓練，
     預測 `months[i : i+RETRAIN_EVERY]`。embargo 隔開訓練尾與測試頭。
  3. **標準化只用當期截面**：產業內 z-score 是逐月橫斷面運算，不跨期。
     刻意不用「全期均值/標準差」——那會把未來的分布洩漏進歷史。

用法
----
    python src/factor_lab.py --compare
        三組（own / reference / all）× 三模型（equal / ridge / lgbm）的績效表

    python src/factor_lab.py --loo --set own
        留一法：逐一拿掉一個因子重跑，IR 反而變高的就是拖累者

    python src/factor_lab.py --greedy --set own --select-span validation --span test
        貪婪前向選擇：在 validation 期挑因子、在 test 期回報成績。
        ⚠️ 不加 --select-span 就是在測試集上挑答案，數字沒有意義

⚠️ 看績效時**別只看 CAGR**。台股 test 期（2020-01 起）是大多頭，等權全樣本
   基準的 CAGR 就有 24%，多頭組合的 30%+ 裡絕大部分是 beta。因子真正的貢獻
   在**超額（Excess）與資訊比率（IR）**——所以 --loo / --greedy 預設用 IR 排序。

    python src/factor_lab.py --cost-scan --set all --model equal,ridge,lgbm
        成本敏感度：同一組分數掃不同的換手成本。換手高的模型這條線會比較陡

    python src/factor_lab.py --factors F-001,F-005,R-003 --model ridge
        只用指定的因子子集跑一次

    --model 預設 equal,ridge（lgbm 需要 pip install lightgbm，用 --model lgbm 開啟）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt
from memory import REFERENCE_PREFIX, Memory, approved_groups

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
ML = CFG.get("ml") or {}

MIN_TRAIN_MONTHS = int(ML.get("min_train_months", 48))
RETRAIN_EVERY = int(ML.get("retrain_every", 12))
EMBARGO = int(ML.get("embargo", 1))
RIDGE_ALPHA = float(ML.get("ridge_alpha", 10.0))
MIN_COVERAGE = float(ML.get("min_feature_coverage", 0.30))

SPANS = {k: (v[0], v[1]) for k, v in CFG["split"].items()}


# ---------------------------------------------------------------------------
# 資料組裝
# ---------------------------------------------------------------------------

def load_panel(which: str = "all", factors: list[str] | None = None
               ) -> tuple[pd.DataFrame, list[str], dict]:
    """
    組出 [stock_id, group, ym, fwd_ret_1m, <因子欄>] 的寬表。

    which: own（F-xxx）/ reference（R-xxx）/ all
    factors: 明確指定要哪些 factor_id（給定時 which 被忽略）
    """
    mem = Memory()
    lib, fv = mem.snapshot()
    if factors:
        keep = [f for f in factors if f in lib]
        missing = [f for f in factors if f not in lib]
        if missing:
            raise SystemExit(f"因子庫裡沒有：{missing}")
    elif which == "own":
        keep = sorted(k for k, v in lib.items() if not v.get("reference") and not k.startswith(REFERENCE_PREFIX))
    elif which == "reference":
        keep = sorted(k for k, v in lib.items() if v.get("reference") or k.startswith(REFERENCE_PREFIX))
    else:
        keep = sorted(lib)
    short_only = [f for f in keep if lib[f].get("trading_use") == "short_only"]
    if short_only and factors:
        raise SystemExit(f"僅做空因子不能加入多頭組合：{short_only}")
    keep = [f for f in keep if f not in short_only]
    if short_only:
        print(f"（多頭組合排除僅做空因子：{short_only}）")
    if not keep:
        raise SystemExit(f"沒有符合的因子（which={which}）")

    fv = fv[fv["factor_id"].isin(keep)]
    wide = fv.pivot_table(index=["ym", "stock_id"], columns="factor_id",
                          values="value", aggfunc="first")

    mb = pd.read_parquet(ROOT / CFG["paths"]["monthly_base"],
                         columns=["stock_id", "ym", "group", "fwd_ret_1m"])
    mb["ym"] = mb["ym"].astype(str)
    mb["stock_id"] = mb["stock_id"].astype(str)
    wide = wide.reset_index()
    wide["ym"] = wide["ym"].astype(str)
    wide["stock_id"] = wide["stock_id"].astype(str)

    df = mb.merge(wide, on=["stock_id", "ym"], how="left")
    for fid in keep:
        groups = approved_groups(lib[fid])
        if groups is not None and fid in df:
            df.loc[~df["group"].isin(groups), fid] = np.nan
    df = df.reset_index(drop=True)  # prediction universe never depends on future labels

    # 覆蓋率太低的因子拿掉——在組合裡它只是噪音來源
    folds = segments(sorted(df.ym.unique()))
    selection_end = min(SPANS["sub_train"][1], folds[0][0]) if folds else SPANS["sub_train"][1]
    historical = df.ym.between(SPANS["sub_train"][0], selection_end)
    feats, dropped = [], []
    for f in keep:
        if f in df.columns and df.loc[historical, f].notna().mean() >= MIN_COVERAGE:
            feats.append(f)
        else:
            cov = df.loc[historical, f].notna().mean() if f in df.columns else 0.0
            dropped.append((f, cov))
    names = {f: lib[f].get("name_zh", f) for f in keep}
    if dropped:
        print(f"（覆蓋率 < {MIN_COVERAGE:.0%} 而排除 {len(dropped)} 個："
              f"{[f'{f}:{c:.0%}' for f, c in dropped[:6]]}…）")
    return df, feats, names


def prep(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """
    產業內逐月 z-score，標籤同樣做產業內去均值。

    ⛔ groupby(["ym","group"]) 是**當期橫斷面**運算，不含任何跨期資訊。
       絕不可改成全期 mean/std——那等於把未來的分布洩漏進歷史。
    """
    out = df.copy()
    g = out.groupby(["ym", "group"])
    for c in feats:
        mu, sd = g[c].transform("mean"), g[c].transform("std")
        out[c] = (out[c] - mu) / sd.replace(0, np.nan)
    out[feats] = out[feats].fillna(0.0)
    out["y"] = out["fwd_ret_1m"] - g["fwd_ret_1m"].transform("mean")
    return out


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

def _fit_predict(kind: str, Xtr, ytr, Xte):
    if kind == "equal":
        return Xte.mean(axis=1)
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=RIDGE_ALPHA).fit(Xtr, ytr).predict(Xte)
    if kind == "lgbm":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03,
                              num_leaves=31, max_depth=5, min_child_samples=200,
                              subsample=0.8, colsample_bytree=0.6,
                              reg_lambda=5.0, random_state=0, n_jobs=-1,
                              verbose=-1)
        return m.fit(Xtr, ytr).predict(Xte)
    raise ValueError(f"未知模型 {kind}")


def segments(months: list[str]) -> list[tuple[str, list[str]]]:
    """
    切出 (訓練窗結尾, 測試月份) 的序列——walk_forward 的唯一切窗來源。

    抽成獨立函式是為了讓前瞻測試能直接檢查切窗本身，而不必從模型輸出反推。
    第 i 段：訓練用 ym <= months[i-1-EMBARGO]，預測 months[i : i+RETRAIN_EVERY]。
    """
    out, i = [], MIN_TRAIN_MONTHS
    while i < len(months):
        out.append((months[i - 1 - EMBARGO], months[i:i + RETRAIN_EVERY]))
        i += RETRAIN_EVERY
    return out


def walk_forward(proc: pd.DataFrame, feats: list[str], kind: str) -> pd.Series:
    """
    walk-forward + embargo 產生樣本外分數。

    embargo 隔開訓練尾與測試頭，避免月頻標籤（fwd_ret_1m 看下一個月）
    讓訓練期最後一個月的標籤與測試期第一個月重疊。
    """
    months = sorted(proc["ym"].unique())
    pred = pd.Series(np.nan, index=proc.index, dtype=float)
    for cut, test_months in segments(months):
        tr = proc[(proc["ym"] <= cut) & proc["y"].notna()]
        te = proc[proc["ym"].isin(test_months)]
        if len(tr) >= 1000 and len(te):
            if kind == "equal":
                pred.loc[te.index] = te[feats].mean(axis=1).values
            else:
                pred.loc[te.index] = _fit_predict(
                    kind, tr[feats].values, tr["y"].values, te[feats].values)
    return pred


def score_and_backtest(proc: pd.DataFrame, feats: list[str], kind: str,
                       span: str = "test") -> dict:
    """跑一次完整流程，回傳指定切分期的績效。"""
    s = walk_forward(proc, feats, kind)
    d = proc[["stock_id", "group", "ym", "fwd_ret_1m"]].copy()
    d["score"] = s
    d = d.dropna(subset=["score"])
    if not len(d):
        return {}
    pf, turn = bt.evaluate(d, span=SPANS.get(span))
    if not pf:                      # 該期不足 6 個月 → perf 回空，數字沒有意義
        return {}
    pf = dict(pf)
    pf["Turnover"] = turn
    pf["NFactors"] = len(feats)
    return pf


# ---------------------------------------------------------------------------
# 三種分析
# ---------------------------------------------------------------------------

def cmd_compare(models: list[str], span: str):
    rows = []
    for which in ("own", "reference", "all"):
        df, feats, _ = load_panel(which)
        if not feats:
            continue
        proc = prep(df, feats)
        for kind in models:
            pf = score_and_backtest(proc, feats, kind, span)
            if pf:
                rows.append({"因子組": which, "模型": kind, **pf})
        print(f"  …{which} 完成（{len(feats)} 個因子）")
    _print_table(rows, f"三組 × {len(models)} 模型（{span} 期）")
    return rows


def cmd_loo(which: str, kind: str, span: str, factors: list[str] | None,
            rank: str = "IR"):
    """留一法：逐一拿掉一個因子重跑；指標變高 = 那個因子在拖累。"""
    df, feats, names = load_panel(which, factors)
    proc = prep(df, feats)
    full = score_and_backtest(proc, feats, kind, span)
    if not full:
        raise SystemExit("全集回測沒有結果（樣本不足？）")
    base = full[rank]
    print(f"\n全集（{len(feats)} 個因子，{kind}，{span} 期）："
          f"{rank} {base:.3f}  超額 {full.get('Excess', float('nan')):.2%}  "
          f"CAGR {full['CAGR']:.2%}  Sharpe {full['Sharpe']:.3f}  "
          f"MaxDD {full['MaxDD']:.2%}  換手 {full['Turnover']:.1%}\n")

    rows = []
    for f in feats:
        sub = [x for x in feats if x != f]
        pf = score_and_backtest(proc, sub, kind, span)
        if not pf:
            continue
        rows.append({"因子": f, "名稱": names.get(f, "")[:16],
                     f"去掉後{rank}": pf[rank],
                     "Δ貢獻": base - pf[rank],
                     "去掉後超額": pf.get("Excess", float("nan")),
                     "去掉後CAGR": pf["CAGR"]})
    rows.sort(key=lambda r: r["Δ貢獻"])
    print(f"{'因子':<8}{'名稱':<18}{'去掉後'+rank:>12}{'Δ貢獻':>9}"
          f"{'去掉後超額':>11}{'去掉後CAGR':>11}")
    print("-" * 70)
    for r in rows:
        flag = "  ← 拖累" if r["Δ貢獻"] < 0 else ""
        print(f"{r['因子']:<8}{r['名稱']:<18}{r['去掉後'+rank]:>12.3f}"
              f"{r['Δ貢獻']:>+9.3f}{r['去掉後超額']:>11.2%}"
              f"{r['去掉後CAGR']:>11.2%}{flag}")
    drags = [r["因子"] for r in rows if r["Δ貢獻"] < 0]
    print(f"\nΔ貢獻 = 全集 {rank} − 去掉該因子後的 {rank}")
    print(f"負值代表「拿掉它反而更好」→ 拖累者 {len(drags)} 個：{drags}")
    return rows


def cmd_greedy(which: str, kind: str, span: str, max_k: int,
               factors: list[str] | None, rank: str = "IR",
               select_span: str | None = None):
    """
    貪婪前向選擇：每輪加入邊際貢獻最大的因子。

    ⚠️ **挑選期與回報期必須分開，否則結果是在測試集上過擬合。**

    walk-forward 保證的是「模型參數」沒看到未來，但「要選哪幾個因子」這個決策
    本身也是一次擬合。如果用 test 期的 IR 來挑，再拿 test 期的 IR 當成績，
    等於拿答案卷去挑答案——k 越大數字越漂亮，但那是選擇偏誤，不是真實 edge。

    `select_span`（`--select-span`）指定在哪一期挑，`span` 指定在哪一期回報。
    兩者不同時，每一步會同時印出「挑選期」與「回報期」的數字：**回報期的曲線
    才是誠實的**，而兩條曲線的落差就是這次選擇偏誤的大小。
    """
    df, feats, names = load_panel(which, factors)
    proc = prep(df, feats)
    sel = select_span or span
    honest = sel != span
    chosen, best, history = [], -np.inf, []
    pool = list(feats)
    if honest:
        print(f"（在 {sel} 期挑選，在 {span} 期回報——回報期的數字才是誠實的）\n")
    for k in range(1, min(max_k, len(feats)) + 1):
        cand_best, cand_pf = None, None
        for f in pool:
            pf = score_and_backtest(proc, chosen + [f], kind, sel)
            if pf and (cand_pf is None or pf[rank] > cand_pf[rank]):
                cand_best, cand_pf = f, pf
        if cand_best is None:
            break
        gain = cand_pf[rank] - (best if np.isfinite(best) else 0.0)
        chosen.append(cand_best)
        pool.remove(cand_best)
        best = cand_pf[rank]
        out = score_and_backtest(proc, chosen, kind, span) if honest else cand_pf
        history.append({"k": k, "加入": cand_best,
                        "名稱": names.get(cand_best, "")[:16],
                        f"{rank}_挑選": best, "邊際": gain,
                        rank: out.get(rank) if out else None,
                        "Excess": out.get("Excess") if out else None,
                        "CAGR": out.get("CAGR") if out else None})
        nan = float("nan")
        if honest:
            tail = (f"｜{span} 期 {rank} {out.get(rank, nan):>6.3f}"
                    f" 超額 {out.get('Excess', nan):>7.2%}")
        else:
            tail = f"  超額 {cand_pf.get('Excess', nan):>7.2%}"
        print(f"  k={k:<3}加入 {cand_best:<8}{names.get(cand_best,'')[:14]:<16}"
              f"{sel} {rank} {best:>6.3f} (邊際 {gain:>+6.3f}){tail}")
    if not honest:
        print("\n⚠️ 挑選與回報用的是同一期，k 越大越好看是**選擇偏誤**，不是 edge。"
              f"\n   要誠實的數字：--greedy --select-span validation --span test")
    # 峰值一律用「挑選期」決定——用回報期挑就又把選擇偏誤放回來了
    hist = [h for h in history if h.get(f"{rank}_挑選") is not None]
    peak = max(hist, key=lambda h: h[f"{rank}_挑選"]) if hist else None
    if peak:
        print(f"\n最佳組合（依 {sel} 期挑）：k={peak['k']}，"
              f"{sel} {rank} {peak[f'{rank}_挑選']:.3f}")
        if honest:
            print(f"  → 同一組在 {span} 期：{rank} {peak[rank]:.3f}"
                  f"，超額 {peak['Excess']:.2%}，CAGR {peak['CAGR']:.2%}")
        print(f"  {chosen[:peak['k']]}")
    return history


def cmd_cost_scan(which: str, models: list[str], span: str,
                  factors: list[str] | None,
                  costs=(0.002, 0.004, 0.008, 0.012, 0.016)):
    """
    成本敏感度：同一組分數，換不同的單次換手成本重跑回測。

    為什麼一定要跑：模型可以靠**更高的換手**換到更好的帳面績效，
    而換手在回測裡只按 `cost` 扣一次固定比例——真實世界的滑點與衝擊成本
    在小型股上遠不是線性的。一個模型如果在成本從 0.4% 拉到 1.2% 時
    績效快速崩掉，代表它的報酬多半來自高頻進出，實盤存疑。

    判讀：看**斜率**而不是絕對值。換手 30% 的模型跟換手 70% 的模型，
    在同一條成本線上的衰退速度差兩倍以上，這件事在單一 cost 的表裡看不到。
    """
    df, feats, _ = load_panel(which, factors)
    proc = prep(df, feats)
    print(f"\n=== 成本敏感度（{which}，{len(feats)} 個因子，{span} 期）===")
    header = "".join(f"{c:>9.1%}" for c in costs)
    print(f"{'模型':<8}{'換手':>7}{header}   ← 超額（年化）")
    print("-" * (15 + 9 * len(costs) + 14))
    rows = []
    for kind in models:
        s = walk_forward(proc, feats, kind)
        d = proc[["stock_id", "group", "ym", "fwd_ret_1m"]].copy()
        d["score"] = s
        d = d.dropna(subset=["score"])
        cells, turn, first = [], None, None
        for c in costs:
            pf, t = bt.evaluate(d, span=SPANS.get(span), cost=c)
            cells.append(pf.get("Excess", float("nan")) if pf else float("nan"))
            turn = t
            first = first if first is not None else cells[0]
        drop = (cells[0] - cells[-1])
        print(f"{kind:<8}{turn:>7.1%}"
              + "".join(f"{v:>9.2%}" for v in cells)
              + f"   衰退 {drop:.2%}")
        rows.append({"模型": kind, "換手": turn,
                     **{f"cost_{c}": v for c, v in zip(costs, cells)},
                     "衰退": drop})
    print("\n判讀：衰退幅度 ≈ 換手率 × (最高成本 − 最低成本) × 12。"
          "\n     明顯超過這個數字代表持股在高成本下被迫改變，模型對成本敏感；"
          "\n     換手越高的模型，這條線越陡——帳面贏但實盤未必。")
    return rows


def _print_table(rows, title):
    if not rows:
        print("（無結果）")
        return
    print(f"\n=== {title} ===")
    print(f"{'因子組':<10}{'模型':<8}{'因子數':>6}{'CAGR':>9}{'Sharpe':>8}"
          f"{'超額':>8}{'IR':>7}{'MaxDD':>9}{'換手':>7}{'月數':>6}")
    print("-" * 78)
    for r in rows:
        print(f"{r['因子組']:<10}{r['模型']:<8}{r['NFactors']:>6}"
              f"{r['CAGR']:>9.2%}{r['Sharpe']:>8.3f}"
              f"{r.get('Excess', float('nan')):>8.2%}{r.get('IR', float('nan')):>7.2f}"
              f"{r['MaxDD']:>9.2%}{r['Turnover']:>7.1%}{r['Months']:>6}")
    bench = next((r.get("BenchCAGR") for r in rows if r.get("BenchCAGR")), None)
    if bench is not None:
        print(f"\n同期等權全樣本基準 CAGR = {bench:.2%}。"
              f"⚠️ 別只看 CAGR——多頭組合的絕對報酬裡大部分是 beta，"
              f"**超額與 IR 才是因子的功勞**。")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="因子組合實驗室")
    ap.add_argument("--compare", action="store_true", help="三組 × 多模型比較")
    ap.add_argument("--loo", action="store_true", help="留一法（找拖累者）")
    ap.add_argument("--greedy", action="store_true", help="貪婪前向選擇")
    ap.add_argument("--cost-scan", action="store_true",
                    help="成本敏感度：同一組分數掃不同的換手成本")
    ap.add_argument("--set", default="all", choices=["own", "reference", "all"],
                    help="因子組（loo/greedy 用）")
    ap.add_argument("--factors", help="逗號分隔的 factor_id，覆寫 --set")
    ap.add_argument("--model", default="equal,ridge",
                    help="模型，逗號分隔：equal / ridge / lgbm")
    ap.add_argument("--span", default="test",
                    choices=["sub_train", "validation", "test"],
                    help="看哪個切分期（預設 test，唯一真正樣本外的）")
    ap.add_argument("--max-k", type=int, default=12, help="貪婪選擇的上限")
    ap.add_argument("--rank", default="IR", choices=["IR", "Sharpe", "Excess"],
                    help="loo/greedy 用哪個指標排序（預設 IR：剔除 beta，"
                         "只看因子的貢獻）")
    ap.add_argument("--select-span", choices=["sub_train", "validation", "test"],
                    help="貪婪選擇在哪一期挑因子（預設與 --span 相同）。"
                         "設成 validation、--span 留 test，才不會在測試集上挑答案")
    ap.add_argument("--save", help="把結果存成 JSON")
    a = ap.parse_args()

    models = [m.strip() for m in a.model.split(",") if m.strip()]
    factors = [f.strip() for f in a.factors.split(",")] if a.factors else None
    if a.span != "test":
        print(f"⚠️ 你選了 {a.span} 期——agent 的因子是用 sub_train+validation 篩出來的，"
              f"在那兩期的績效是 in-sample，會虛高。真正的樣本外只有 test。\n")

    if a.compare:
        out = cmd_compare(models, a.span)
    elif a.cost_scan:
        out = cmd_cost_scan(a.set, models, a.span, factors)
    elif a.loo:
        out = cmd_loo(a.set, models[0], a.span, factors, a.rank)
    elif a.greedy:
        out = cmd_greedy(a.set, models[0], a.span, a.max_k, factors, a.rank,
                         a.select_span)
    else:
        df, feats, _ = load_panel(a.set, factors)
        proc = prep(df, feats)
        out = [{"因子組": a.set if not factors else "custom", "模型": k,
                **score_and_backtest(proc, feats, k, a.span)} for k in models]
        out = [r for r in out if r.get("Sharpe") is not None]
        _print_table(out, f"{a.set if not factors else factors}（{a.span} 期）")

    if a.save:
        Path(a.save).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                           default=float), encoding="utf-8")
        print(f"\n已存 {a.save}")


if __name__ == "__main__":
    main()
