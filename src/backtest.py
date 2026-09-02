"""
回測核心：由分數（score）建構產業內多頭組合，計算月報酬、換手與績效。

移植自前置專案 tw_alpha_strategy/src/backtest.py 的 portfolio_returns / perf，
改為讀 config.yaml（不再有模組層可變全域），並補上「指定期間」的切片。
移植而非重寫：那套邏輯已經驗證過，重寫只會製造語意漂移。

口徑與 agent 的評估一致：
  - 逐月「在各產業內」選分數前 top_q，與 eval_candidates 的產業內 IC 同口徑
  - 產業內不足 min_stocks_per_group 檔就跳過該產業該月
  - 換手 = |本期持股 △ 上期持股| / |聯集|，成本按換手比例扣在報酬上

⛔ 前瞻紀律：本模組只吃已經算好的 score 與 fwd_ret_1m，不做任何跨期運算。
   score 的無前瞻性由上游（factor_lab 的 walk-forward + embargo）保證。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
BT = CFG.get("backtest") or {}

TOP_Q = float(BT.get("top_q", 0.10))
COST = float(BT.get("cost", 0.004))
WEIGHTING = str(BT.get("weighting", "equal"))
MIN_STOCKS = int(BT.get("min_stocks_per_group",
                        CFG["funnel"].get("min_stocks_per_group", 8)))
ANN = int(BT.get("ann", 12))

REQUIRED = ("stock_id", "group", "ym", "score", "fwd_ret_1m")


def portfolio_returns(df: pd.DataFrame, top_q: float = TOP_Q,
                      weighting: str = WEIGHTING, cost: float = COST,
                      min_stocks: int = MIN_STOCKS) -> pd.DataFrame:
    """
    逐月在各產業內取分數前 top_q 做多、後 top_q 作為空頭參考。

    回傳 index=ym 的 DataFrame：
      long        多頭月報酬（已扣換手成本）
      long_short  多空月報酬（已扣換手成本；台股難放空，僅供參考）
      benchmark   當月全樣本等權報酬
      turnover    換手比例
      nhold       持股數
    """
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"portfolio_returns 需要欄位 {miss}")

    rows, prev = {}, set()
    for ym, g in df.groupby("ym", sort=True):
        g = g.dropna(subset=["score", "fwd_ret_1m"])
        if len(g) < 20:
            continue
        longs, shorts = [], []
        for _, gg in g.groupby("group"):
            if len(gg) < min_stocks:
                continue
            n = max(1, int(len(gg) * top_q))
            ranked = gg.sort_values("score", ascending=False)
            longs.append(ranked.head(n))
            shorts.append(ranked.tail(n))
        if not longs:
            continue
        L, S = pd.concat(longs), pd.concat(shorts)

        if weighting == "score":
            w = L["score"].rank()
            w = w / w.sum()
        else:                                  # equal
            w = pd.Series(1.0 / len(L), index=L.index)

        cur = set(L["stock_id"])
        turn = 1.0 if not prev else len(cur ^ prev) / max(len(cur | prev), 1)
        prev = cur

        long_ret = float((L["fwd_ret_1m"] * w).sum())
        short_ret = float(S["fwd_ret_1m"].mean())
        rows[ym] = {
            "long": long_ret - cost * turn,
            "long_short": (long_ret - short_ret) - cost * turn,
            "benchmark": float(g["fwd_ret_1m"].mean()),
            "turnover": turn,
            "nhold": len(L),
        }
    return pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame(
        columns=["long", "long_short", "benchmark", "turnover", "nhold"])


def perf(r: pd.Series, ann: int = ANN) -> dict:
    """年化績效。樣本 < 6 個月回空 dict（數字沒有意義）。"""
    r = pd.Series(r).dropna().astype(float)
    if len(r) < 6:
        return {}
    curve = (1 + r).cumprod()
    vol = r.std() * np.sqrt(ann)
    return {
        "CAGR": float(curve.iloc[-1] ** (ann / len(r)) - 1),
        "Vol": float(vol),
        "Sharpe": float((r.mean() * ann) / vol) if vol > 0 else np.nan,
        "MaxDD": float((curve / curve.cummax() - 1).min()),
        "WinRate": float((r > 0).mean()),
        "Months": int(len(r)),
    }


def slice_span(rets: pd.DataFrame, span: tuple[str, str] | None) -> pd.DataFrame:
    """依 (起, 迄) 的 ym 字串切片；None = 全期。"""
    if span is None or not len(rets):
        return rets
    lo, hi = span
    idx = [m for m in rets.index if lo <= str(m) <= hi]
    return rets.loc[idx]


def evaluate(df: pd.DataFrame, span: tuple[str, str] | None = None,
             **kw) -> tuple[dict, float]:
    """
    跑一次回測並回傳 (績效 dict, 平均換手)。span 用來只看某個切分期。

    績效 dict 內含超額欄位（Excess / IR / BenchCAGR）——**光看 CAGR 會誤判**：
    台股 test 期（2020-01 起）本身就是大多頭，等權全樣本基準的 CAGR 就有 24%，
    多頭組合的 30%+ 裡絕大部分是 beta，不是因子的功勞。
    """
    rets = slice_span(portfolio_returns(df, **kw), span)
    if not len(rets):
        return {}, float("nan")
    pf = perf(rets["long"])
    if pf:
        pf.update(excess_stats(rets, kw.get("ann", ANN)))
    return pf, float(rets["turnover"].mean())


def excess_vs_benchmark(rets: pd.DataFrame) -> pd.Series:
    """多頭相對當月等權基準的超額報酬。"""
    return rets["long"] - rets["benchmark"]


def excess_stats(rets: pd.DataFrame, ann: int = ANN) -> dict:
    """超額報酬的年化值、資訊比率（IR）、勝月比例，與基準本身的 CAGR。"""
    ex = excess_vs_benchmark(rets).dropna()
    if len(ex) < 6:
        return {}
    sd = ex.std() * np.sqrt(ann)
    bench = perf(rets["benchmark"], ann)
    return {
        "Excess": float(ex.mean() * ann),
        "IR": float((ex.mean() * ann) / sd) if sd > 0 else np.nan,
        "ExcessWin": float((ex > 0).mean()),
        "BenchCAGR": float(bench.get("CAGR", np.nan)),
    }
