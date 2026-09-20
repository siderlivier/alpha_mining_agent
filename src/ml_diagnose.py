"""
合成分數的「這個 IR 是真的嗎」診斷套件。

`factor_lab.py --compare` 會告訴你 LightGBM 的 IR 有 2.57。這支模組的存在
理由只有一個：**一個漂亮的 IR 有很多種假法**，而每一種假法要用不同的方法拆穿。

    假法                          拆穿的方法              指令
    ─────────────────────────────────────────────────────────────
    只是搭上多頭順風車            分多空 régime 看        --regime
    只是少數幾個月撐起來的        逐月攤開 + 拿掉最好N月   --breadth
    只是小型股/低流動股溢酬       流動性歸因與過濾        --liquidity
    只是規模因子換個名字          規模中性版重跑          --importance
    只是試了很多策略挑到的幸運兒  Deflated Sharpe / PBO   --dsr
    報酬其實被換手成本吃光        成本敏感度              factor_lab --cost-scan

移植自前置專案 tw_alpha_strategy/src/ml/ 的四支腳本（ml_model 的重要性與
規模中性、ml_regime、ml_liquidity、ml_validation），改成**完全在本專案內**
執行：因子值讀 memory/factor_values.parquet、標籤與流動性讀
data/monthly_base.parquet，不 import 前置專案。

三處刻意偏離原版：

1. **切分期紀律**：原版把全部樣本外月份混在一起看。本專案有 sub_train /
   validation / test 三段，agent 的因子是用前兩段選出來的，所以一律只看
   `test` 期（`--span` 可改，會跳警告）。
2. **規模代理用相關度偵測，不用寫死的名單**。原版寫死
   `SIZE_FEATURES = ["liq_amt", "neg_size"]`，那只涵蓋得了 DFS 因子；
   agent 自己挖的因子也可能是規模的變體，寫死的名單抓不到。改成量測每個
   因子與 log(成交金額) 的橫斷面排名相關，超過門檻就算規模代理。
3. **成本用參數傳，不改模組全域**。原版 `bt.COST = cst` 直接改全域再改回來，
   一旦中途丟例外就會留下污染的全域狀態。

⛔ 前瞻紀律：本模組只消費 `factor_lab.walk_forward()` 產生的樣本外分數，
   自己不做任何跨期運算。流動性百分位是「當月橫斷面排名」，
   `amt_21` 本身是過去 21 個交易日的均值（往回看）。

用法
----
    python src/ml_diagnose.py --all                    # 五項全跑（不含 --timing）
    python src/ml_diagnose.py --regime                 # 多空 régime 分解
    python src/ml_diagnose.py --breadth                # 逐月分布、集中度、產業組成
    python src/ml_diagnose.py --liquidity              # 流動性歸因 + 過濾×成本
    python src/ml_diagnose.py --importance             # 特徵重要性 + SHAP + 規模中性
    python src/ml_diagnose.py --dsr                    # Deflated Sharpe + PBO
    python src/ml_diagnose.py --regime-factors         # 逐因子的多空月 ICIR 差異
    python src/ml_diagnose.py --timing                 # 擇時上限與損益兩平準確率
    python src/ml_diagnose.py --all --set own --model ridge

規劃狀態模型（HMM 等）之前，先跑 --timing 與 --regime-factors：
前者告訴你「用狀態決定進出場」的門檻有多高（實測要 80% 準確率才損益兩平），
後者告訴你「用狀態決定因子權重」有多少空間（實測 test 期有 12/51 個因子
在多空月變號）。後者的失敗代價低一個量級，是比較務實的接法。

需要：scipy（--dsr）、lightgbm（--importance / --model lgbm）、shap（選配）
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt
import factor_lab as fl

ROOT = Path(__file__).resolve().parents[1]
ANN = bt.ANN
NAMES_PATH = ROOT / "data" / "stock_names.json"

# 規模代理的判定門檻：與 log(成交金額) 的橫斷面排名相關度
SIZE_CORR_THRESHOLD = 0.60
# 公式裡出現這些欄位 = 定義上就是流動性／規模代理，不必等相關度達標
SIZE_FIELDS = ("amt_21", "turn_21")


# ---------------------------------------------------------------------------
# 共用：算分數、切期、組報酬
# ---------------------------------------------------------------------------

def build_scores(which: str, kind: str, factors: list[str] | None = None):
    """跑一次 walk-forward，回傳 (帶 score 的面板, 特徵清單, 中文名對照)。"""
    df, feats, names = fl.load_panel(which, factors)
    proc = fl.prep(df, feats)
    proc["score"] = fl.walk_forward(proc, feats, kind)
    out = proc.dropna(subset=["score"]).copy()
    if not len(out):
        raise SystemExit("沒有任何樣本外分數（暖身期太長？資料太短？）")
    return out, feats, names, proc


def attach_liquidity(d: pd.DataFrame) -> pd.DataFrame:
    """
    併上 amt_21 並算「當月橫斷面流動性百分位」。

    amt_21 = 過去 21 個交易日的平均成交金額，本身就是往回看的；
    百分位是當月橫斷面排名，兩者都不含未來資訊。
    """
    mb = pd.read_parquet(ROOT / fl.CFG["paths"]["monthly_base"],
                         columns=["stock_id", "ym", "amt_21"])
    mb["ym"] = mb["ym"].astype(str)
    mb["stock_id"] = mb["stock_id"].astype(str)
    out = d.merge(mb, on=["stock_id", "ym"], how="left")
    out = out.dropna(subset=["amt_21"])
    out["liq_pct"] = out.groupby("ym")["amt_21"].rank(pct=True)
    return out


def span_mask(d: pd.DataFrame, span: str) -> pd.DataFrame:
    lo, hi = fl.SPANS[span]
    return d[(d["ym"] >= lo) & (d["ym"] <= hi)]


def rets_of(d: pd.DataFrame, span: str | None = None, **kw) -> pd.DataFrame:
    """由帶 score 的面板算組合月報酬，並切到指定期間。"""
    cols = ["stock_id", "group", "ym", "fwd_ret_1m", "score"]
    r = bt.portfolio_returns(d[cols], **kw)
    return bt.slice_span(r, fl.SPANS.get(span)) if span else r


def excess_stats(rets: pd.DataFrame) -> dict:
    """年化超額、IR、t 值、勝率。t 用來判斷超額是不是統計上站得住。"""
    ex = bt.excess_vs_benchmark(rets).dropna()
    n = len(ex)
    if n < 4:
        return {}
    sd = ex.std()
    return {"月數": n, "年化超額": ex.mean() * ANN,
            "IR": (ex.mean() * ANN) / (sd * np.sqrt(ANN)) if sd > 0 else np.nan,
            "t": ex.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan,
            "勝率": float((ex > 0).mean())}


# ---------------------------------------------------------------------------
# 1. régime 分解
# ---------------------------------------------------------------------------

def cmd_regime(which: str, kind: str, span: str, factors):
    """
    把樣本外月份依「大盤（全池等權基準）漲跌」分組，各自看超額。

    要回答的問題：IR 2.57 是不是只因為 test 期剛好是大多頭？
    **如果大盤下跌月的超額仍為正且 t 值不小，那就是真的選股技巧**；
    若下跌月超額大幅轉負，代表高 IR 相當程度是多頭順風車。
    """
    d, feats, _, _ = build_scores(which, kind, factors)
    d = attach_liquidity(d)

    for label, sub in (("全體股票池", d),
                       ("可交易版（最流動 30%）", d[d["liq_pct"] >= 0.7])):
        rets = rets_of(sub, span)
        if not len(rets):
            continue
        ex = bt.excess_vs_benchmark(rets).dropna()
        bm = rets["benchmark"].reindex(ex.index)
        q25 = bm.quantile(0.25)
        buckets = [("全部月份", ex.index),
                   ("大盤上漲月", bm[bm > 0].index),
                   ("大盤下跌月", bm[bm <= 0].index),
                   ("最差 25% 月", bm[bm <= q25].index)]
        print(f"\n===== {label}（{which}/{kind}，{span} 期，{len(feats)} 因子）=====")
        print(f"{'狀態':>12} {'月數':>5} {'年化超額':>10} {'IR':>7} {'t':>7} {'勝率':>7}")
        for lab, idx in buckets:
            s = excess_stats(rets.reindex(idx).dropna(how="all"))
            if s:
                print(f"{lab:>12} {s['月數']:>5d} {s['年化超額']:>10.2%} "
                      f"{s['IR']:>7.2f} {s['t']:>7.2f} {s['勝率']:>7.1%}")
        years = [str(m)[:4] for m in ex.index]
        annual = [(y, (1 + g).prod() - 1) for y, g in ex.groupby(years)]
        print("  逐年超額：" + "　".join(f"{y}:{v:+.1%}" for y, v in annual))

    print("\n判讀：大盤下跌月／最差 25% 月的超額若仍為正且 t 不小 → 逆風時也有效，"
          "\n     不是多頭 régime 的順風車；若下跌月大幅轉負 → 高 IR 靠的是 beta。")


# ---------------------------------------------------------------------------
# 1b. 廣度：超額是「每個月都有一點」還是「少數幾個月撐起來的」？
# ---------------------------------------------------------------------------

def cmd_breadth(which: str, kind: str, span: str, factors, drop_n: int = 3):
    """
    régime 表只給每個桶的平均，看不出那個平均是怎麼來的。

    「大盤下跌月平均超額 +13%」有兩種完全不同的可能：
      (a) 27 個月裡有 23 個月都 +1% 上下  → 穩定的選股技巧
      (b) 24 個月接近 0、3 個月 +10%      → 少數事件，下次未必再有

    這支就是把逐月數字攤開來，並做「拿掉最好的 N 個月」的穩健度檢查。
    順便印出持股的產業組成——因為產業配置在本策略裡是**結構性鎖定**的
    （逐月在各產業內取前 top_q），看一眼就知道有沒有跑掉。
    """
    d, feats, _, _ = build_scores(which, kind, factors)
    d = attach_liquidity(d)

    for label, sub in (("全體股票池", d),
                       ("可交易版（最流動 30%）", d[d["liq_pct"] >= 0.7])):
        rets = rets_of(sub, span)
        if not len(rets):
            continue
        ex = bt.excess_vs_benchmark(rets).dropna()
        bm = rets["benchmark"].reindex(ex.index)
        print(f"\n===== {label}（{which}/{kind}，{span} 期）=====")
        print(f"{'桶':<12}{'月數':>5}{'超額為正':>9}{'中位超額':>10}"
              f"{'年化超額':>10}{'IR':>7}{'去掉最好'+str(drop_n)+'月的IR':>16}")
        for lab, idx in (("全部月份", ex.index),
                         ("大盤上漲月", bm[bm > 0].index),
                         ("大盤下跌月", bm[bm <= 0].index)):
            e = ex.reindex(idx).dropna()
            if len(e) < 4:
                continue
            rest = e.drop(e.nlargest(min(drop_n, len(e) - 2)).index)
            ir = e.mean() * ANN / (e.std() * np.sqrt(ANN)) if e.std() > 0 else np.nan
            ir2 = (rest.mean() * ANN / (rest.std() * np.sqrt(ANN))
                   if rest.std() > 0 else np.nan)
            print(f"{lab:<12}{len(e):>5}{(e > 0).mean():>9.0%}{e.median():>10.2%}"
                  f"{e.mean() * ANN:>10.2%}{ir:>7.2f}{ir2:>16.2f}")
        dn = ex.reindex(bm[bm <= 0].index).dropna()
        if len(dn) > drop_n:
            share = dn.nlargest(drop_n).sum() / dn.sum() if dn.sum() != 0 else np.nan
            print(f"\n  下跌月超額的集中度：最大 {drop_n} 個月佔總超額 {share:.0%}"
                  f"（{drop_n}/{len(dn)} = {drop_n / len(dn):.0%} 的月份）")
            print(f"  → 佔比若遠高於月份佔比，代表是少數事件撐起來的。")
            print(f"\n  下跌月逐月（大盤 / 超額）：")
            for m in dn.sort_values().index:
                print(f"    {m}  大盤 {bm[m]:>+8.2%}   超額 {dn[m]:>+8.2%}")

    # 產業組成
    ds = span_mask(d, span)
    H = _holdings(ds)
    uni = ds.groupby("group").size() / len(ds)
    hs = H.groupby("group").size() / len(H)
    cs = H.groupby("group")["contrib"].sum()
    cs = cs / cs.sum()
    print(f"\n===== 持股的產業組成 vs 全池 =====")
    print(f"{'產業':<8}{'持股佔比':>10}{'全池佔比':>10}{'偏離':>8}"
          f"{'貢獻佔比':>10}{'平均月報酬':>12}")
    for g in hs.sort_values(ascending=False).index:
        print(f"{g:<8}{hs[g]:>10.1%}{uni.get(g, 0):>10.1%}"
              f"{hs[g] - uni.get(g, 0):>+8.1%}{cs[g]:>10.1%}"
              f"{H[H['group'] == g]['fwd_ret_1m'].mean():>12.2%}")
    print("\n⚠️ 產業佔比與全池幾乎相同是**設計使然**，不是巧合：回測是逐月"
          "\n   『在各產業內』取分數前 top_q，產業權重因此結構性地鎖在全池比例上。"
          "\n   換句話說本策略**完全沒有產業輪動、也沒有市場擇時**——"
          "\n   全部的超額都來自產業內的選股。")


def factor_ic_by_regime(proc: pd.DataFrame, feats: list[str], span_lo: str,
                        span_hi: str, names: dict) -> pd.DataFrame:
    """逐因子算「大盤上漲月 / 下跌月」的 ICIR，並回傳差異。"""
    p = proc[(proc["ym"] >= span_lo) & (proc["ym"] <= span_hi)]
    mkt = p.groupby("ym")["fwd_ret_1m"].mean()      # 全池等權 = 大盤代理
    up, dn = set(mkt[mkt > 0].index), set(mkt[mkt <= 0].index)

    def _icir(s):
        return s.mean() / s.std() if len(s) > 3 and s.std() > 0 else np.nan

    rows = []
    for f in feats:
        ic = {}
        for ym, g in p.groupby("ym"):
            per = []
            for grp, gg in g.groupby("group"):
                s = gg[[f, "fwd_ret_1m"]].dropna()
                if len(s) < bt.MIN_STOCKS or s[f].nunique() < 2:
                    continue
                v = s[f].rank().corr(s["fwd_ret_1m"].rank())
                if pd.notna(v):
                    per.append(v)
            if per:
                ic[ym] = float(np.mean(per))
        ic = pd.Series(ic)
        if len(ic) < 30:
            continue
        a, b = _icir(ic[ic.index.isin(up)]), _icir(ic[ic.index.isin(dn)])
        if pd.isna(a) or pd.isna(b):
            continue
        rows.append({"因子": f, "名稱": (names.get(f) or "")[:14],
                     "ICIR_全": _icir(ic), "ICIR_漲": a, "ICIR_跌": b,
                     "差異": b - a})
    return pd.DataFrame(rows)


def cmd_regime_factors(which: str, span: str, factors, threshold: float = 0.3):
    """
    逐因子的「上漲月 vs 下跌月」ICIR 差異——狀態模型（HMM 等）真正該接的地方。

    為什麼這比「預測漲跌決定進出場」重要得多：

      進出場是**二元賭注**，猜錯一個上漲月就整個月踏空，所以損益兩平的
      準確率高達 80%（見 --timing）。但如果狀態訊號拿來**切換因子權重**，
      猜錯只是用了比較不合適的那組因子——它們的 IC 仍然是正的，
      損失是「少賺一點」而不是「整個月不在場」。**失敗的代價低一個量級。**

    分類（哪些因子偏防禦）**只能用 test 期以外的資料決定**，否則就是
    拿答案卷分類，跟 --greedy 那個陷阱是同一回事。
    """
    df, feats, names = fl.load_panel(which, factors)
    proc = fl.prep(df, feats)

    cls_hi = fl.SPANS["validation"][1]
    print(f"\n=== 因子的 ICIR：大盤上漲月 vs 下跌月 ===")
    for label, (lo, hi) in (("分類期（≤validation，可用來分類）",
                             (fl.SPANS["sub_train"][0], cls_hi)),
                            (f"{span} 期（只驗證，不可用來分類）",
                             fl.SPANS[span])):
        R = factor_ic_by_regime(proc, feats, lo, hi, names)
        if not len(R):
            continue
        R = R.reindex(R["差異"].abs().sort_values(ascending=False).index)
        print(f"\n----- {label}：{lo} ~ {hi} -----")
        print(f"{'因子':<8}{'名稱':<16}{'ICIR_全':>9}{'ICIR_漲':>9}"
              f"{'ICIR_跌':>9}{'差異(跌-漲)':>12}")
        for _, r in R.head(10).iterrows():
            print(f"{r['因子']:<8}{r['名稱']:<16}{r['ICIR_全']:>9.2f}"
                  f"{r['ICIR_漲']:>9.2f}{r['ICIR_跌']:>9.2f}{r['差異']:>12.2f}")
        n = len(R)
        print(f"  |差異| > 0.5：{(R['差異'].abs() > 0.5).sum()}/{n}　"
              f"|差異| > 1.0：{(R['差異'].abs() > 1.0).sum()}/{n}　"
              f"漲跌變號：{((R['ICIR_漲'] * R['ICIR_跌']) < 0).sum()}/{n}")
        if label.startswith("分類期"):
            defensive = list(R[R["差異"] > threshold]["因子"])
            offensive = list(R[R["差異"] < -threshold]["因子"])
            print(f"\n  依分類期判定（門檻 ±{threshold}）：")
            print(f"    防禦型（下跌月較強）{len(defensive)} 個：{defensive}")
            print(f"    攻擊型（上漲月較強）{len(offensive)} 個：{offensive}")
            print(f"    中性 {n - len(defensive) - len(offensive)} 個")

    print("\n判讀：變號的因子越多，狀態訊號的價值越高——那代表同一個因子在兩種"
          "\n     狀態下的方向相反，混在一起用等於互相抵銷。")
    print("\n⚠️ 分類必須只用分類期的資料。用 test 期的差異來分類、再回頭在 test 期"
          "\n   驗證，是與 --greedy 同一種選擇偏誤，數字會漂亮但不可複製。")


def _holdings(d: pd.DataFrame) -> pd.DataFrame:
    """重建逐月持股表（與 backtest.portfolio_returns 同口徑）。"""
    rows = []
    for ym, g in d.groupby("ym"):
        picks = []
        for grp, gg in g.groupby("group"):
            if len(gg) < bt.MIN_STOCKS:
                continue
            n = max(1, int(len(gg) * bt.TOP_Q))
            picks.append(gg.sort_values("score", ascending=False).head(n))
        if not picks:
            continue
        h = pd.concat(picks).copy()
        h["w"] = 1.0 / len(h)
        h["contrib"] = h["w"] * h["fwd_ret_1m"]
        rows.append(h)
    return pd.concat(rows)


# ---------------------------------------------------------------------------
# 1c. 擇時研究：加一層市場方向判斷值不值得？
# ---------------------------------------------------------------------------

def cmd_timing(which: str, kind: str, span: str, factors, trials: int = 1000):
    """
    在動手寫擇時模組（HMM 之類）之前，先算清楚兩件事：

      1. **上限**：完美預測市場方向能拿到多少？
      2. **損益兩平的準確率**：要多準才不會反而更差？

    第 2 點是關鍵而且反直覺。台股 test 期有 65% 的月份是上漲的，
    所以「永遠猜漲」這個零成本策略就有 65% 準確率，而且它的績效**就是**
    目前滿倉的績效。擇時模組必須明顯贏過 65% 這個門檻才有價值——
    猜錯一個上漲月的代價，遠大於猜對一個下跌月的收益。
    """
    d, feats, _, _ = build_scores(which, kind, factors)
    r = rets_of(d, span)
    if len(r) < 24:
        raise SystemExit(f"{span} 期只有 {len(r)} 個月，不足以做擇時研究")

    print(f"\n===== 目前回測算出來的三條腿（{span} 期，{len(r)} 個月）=====")
    for col, desc in (("long", "只做多（factor_lab / ml_diagnose 全部用這條）"),
                      ("long_short", "多空對沖（算出來但目前沒人讀）"),
                      ("benchmark", "全池等權基準")):
        p = bt.perf(r[col])
        print(f"{col:<12}CAGR {p['CAGR']:>8.2%}  Sharpe {p['Sharpe']:>6.2f}  "
              f"MaxDD {p['MaxDD']:>8.2%}  勝率 {p['WinRate']:>6.1%}   {desc}")

    short_leg = r["long_short"] - r["long"]      # = −(後 top_q 的報酬)
    print(f"\n空頭腿單獨：年化 {short_leg.mean() * ANN:+.2%}，"
          f"月勝率 {(short_leg > 0).mean():.1%}")
    print("  → 為負代表「後段班股票平均仍在漲」。多頭市場裡放空是逆風，"
          "\n    空頭腿的價值在**對沖 beta**（看 long_short 的 Sharpe 與 MaxDD），不在賺錢。")

    bmk = r["benchmark"]
    up = bmk > 0
    base = bt.perf(r["long"])
    print(f"\n===== 擇時的上限（完美預測，不可能達成，只當天花板）=====")
    for lab, series in (
            ("下跌月空手", r["long"].where(up, 0.0)),
            ("下跌月反手做空後段班", r["long"].where(up, short_leg))):
        p = bt.perf(series)
        print(f"  {lab:<22}CAGR {p['CAGR']:>8.2%}  Sharpe {p['Sharpe']:>6.2f}  "
              f"MaxDD {p['MaxDD']:>8.2%}")
    print(f"  {'（對照）一直滿倉':<22}CAGR {base['CAGR']:>8.2%}  "
          f"Sharpe {base['Sharpe']:>6.2f}  MaxDD {base['MaxDD']:>8.2%}")

    up_rate = float(up.mean())
    print(f"\n===== 損益兩平的準確率（下跌月空手，{trials} 次模擬取中位）=====")
    print(f"  基準線：{span} 期有 {up_rate:.0%} 的月份上漲，"
          f"「永遠猜漲」就有 {up_rate:.0%} 準確率且績效等同滿倉。")
    print(f"\n{'準確率':>8}{'中位CAGR':>11}{'中位Sharpe':>12}{'勝過滿倉的機率':>16}")
    rng = np.random.default_rng(0)
    truth = up.values
    rows = []
    for acc in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00):
        cagrs, shs, wins = [], [], 0
        for _ in range(trials):
            pred = np.where(rng.random(len(truth)) < acc, truth, ~truth)
            p = bt.perf(r["long"].where(pd.Series(pred, index=r.index), 0.0))
            cagrs.append(p["CAGR"])
            shs.append(p["Sharpe"])
            wins += p["CAGR"] > base["CAGR"]
        rows.append({"準確率": acc, "CAGR": float(np.median(cagrs)),
                     "Sharpe": float(np.median(shs)), "勝率": wins / trials})
        print(f"{acc:>8.0%}{np.median(cagrs):>11.2%}{np.median(shs):>12.2f}"
              f"{wins / trials:>16.1%}")
    print(f"\n（滿倉基準 CAGR {base['CAGR']:.2%}，Sharpe {base['Sharpe']:.2f}）")
    print("\n⚠️ 這張表對擇時模組**有利**，實際門檻只會更高，因為模擬假設：")
    print("   (a) 準確率與月份漲跌幅無關——實際模型常在小波動上對、大波動上錯")
    print("   (b) 進出場沒有成本——空手再進場是一次 100% 換手")
    print("   (c) 空手期間報酬為 0——沒有計入資金閒置的機會成本")
    return rows


# ---------------------------------------------------------------------------
# 2. 流動性歸因與壓力測試
# ---------------------------------------------------------------------------

def cmd_liquidity(which: str, kind: str, span: str, factors):
    """
    (A) 持股的流動性分布與各層報酬貢獻；(B) 流動性過濾 × 成本壓力測試。

    要回答的問題：這個 IR 有多少來自「帳面上買得到、實際上買不到」的小型股？
    """
    d, feats, _, _ = build_scores(which, kind, factors)
    d = attach_liquidity(d)
    d = span_mask(d, span)
    if not len(d):
        raise SystemExit(f"{span} 期沒有資料")

    top_q = bt.TOP_Q
    holds = []
    for ym, g in d.groupby("ym"):
        picks = []
        for grp, gg in g.groupby("group"):
            if len(gg) < bt.MIN_STOCKS:
                continue
            n = max(1, int(len(gg) * top_q))
            picks.append(gg.sort_values("score", ascending=False).head(n))
        if not picks:
            continue
        h = pd.concat(picks).copy()
        h["w"] = 1.0 / len(h)
        h["contrib"] = h["w"] * h["fwd_ret_1m"]
        holds.append(h)
    H = pd.concat(holds)

    print(f"\n================ (A) 流動性歸因（{which}/{kind}，{span} 期）================")
    print(f"持股的流動性百分位平均：{H['liq_pct'].mean():.2f}"
          f"（0.5 = 市場中位；越低 = 越偏小型／低流動）")
    H["tier"] = pd.cut(H["liq_pct"], [0, 1/3, 2/3, 1.0],
                       labels=["低流動", "中流動", "高流動"], include_lowest=True)
    tot = H.groupby("tier", observed=True)["contrib"].sum()
    cnt = H.groupby("tier", observed=True).size()
    print(f"\n{'層級':>6} {'持股次數':>9} {'貢獻佔比':>9} {'平均月報酬':>11}")
    for t in ["低流動", "中流動", "高流動"]:
        if t not in cnt:
            continue
        print(f"{t:>6} {cnt[t]:>9} {tot[t]/tot.sum():>9.1%} "
              f"{H[H['tier'] == t]['fwd_ret_1m'].mean():>11.2%}")
    print("→ 低流動層貢獻佔比若遠高於其持股佔比，代表 edge 大半來自難交易的小型股。")

    name_map = json.loads(NAMES_PATH.read_text(encoding="utf-8")) \
        if NAMES_PATH.exists() else {}
    agg = H.groupby("stock_id").agg(total_contrib=("contrib", "sum"),
                                    avg_liqpct=("liq_pct", "mean"),
                                    months=("contrib", "size"))
    agg["name"] = agg.index.map(lambda s: name_map.get(str(s), "?"))
    print("\n貢獻最大的 15 檔：")
    print(agg.sort_values("total_contrib", ascending=False).head(15)
          [["name", "total_contrib", "avg_liqpct", "months"]]
          .to_string(formatters={"total_contrib": "{:+.3f}".format,
                                 "avg_liqpct": "{:.2f}".format}))

    print(f"\n============ (B) 流動性過濾 × 成本壓力測試 ============")
    print(f"{'剔除底X%':>9} {'持股~':>6} {'CAGR@0.4%':>10} {'Sharpe':>7} "
          f"{'年化超額':>9} {'超額t':>7} {'換手':>6} {'CAGR@0.8%':>10} {'CAGR@1.2%':>10}")
    print("-" * 82)
    rows = []
    for drop in (0.0, 0.3, 0.5, 0.7):
        sub = d if drop <= 0 else d[d["liq_pct"] >= drop]
        res = {c: rets_of(sub, span, cost=c) for c in (0.004, 0.008, 0.012)}
        r0 = res[0.004]
        if not len(r0):
            continue
        pf = bt.perf(r0["long"])
        st = excess_stats(r0)
        if not pf or not st:
            continue
        print(f"{drop:>9.0%} {int(r0['nhold'].mean()):>6d} {pf['CAGR']:>10.2%} "
              f"{pf['Sharpe']:>7.2f} {st['年化超額']:>9.2%} {st['t']:>7.2f} "
              f"{r0['turnover'].mean():>6.0%} "
              f"{bt.perf(res[0.008]['long'])['CAGR']:>10.2%} "
              f"{bt.perf(res[0.012]['long'])['CAGR']:>10.2%}")
        rows.append({"剔除": drop, **pf, **st})
    print("\n判讀：剔除越多低流動股 → 超額／Sharpe 若快速下滑，代表 IR 大半是小型股假象；"
          "\n     若過濾到剔除底 70% 仍穩健，代表這個 edge 在買得到的股票上也成立。")
    return rows


# ---------------------------------------------------------------------------
# 3. 特徵重要性 + SHAP + 規模中性
# ---------------------------------------------------------------------------

def formula_size_proxies(names_lib: dict, feats: list[str]) -> list[str]:
    """
    公式裡**直接用到**流動性／周轉欄位的因子——這是定義上的規模代理，
    不需要量測相關度就成立。

    為什麼要有這一道：實測發現 agent 挖出的「冷門股」系列因子
    （F-013 `sub(cs_rank(rev_yoy), cs_rank(amt_21))`、
      F-020 `mul(cs_rank(op_margin), cs_rank(neg(amt_21)))`）
    公式裡明明白白寫著 `amt_21`，但量測到的 ρ 只有 −0.58 / −0.46，
    低於 0.60 的門檻而漏網。**相關度是行為證據，公式是定義證據**，
    兩道都要。
    """
    hits = []
    for f in feats:
        formula = (names_lib.get(f) or {}).get("formula") or ""
        if any(fld in formula for fld in SIZE_FIELDS):
            hits.append(f)
    return hits


def detect_size_proxies(proc: pd.DataFrame, feats: list[str],
                        threshold: float = SIZE_CORR_THRESHOLD):
    """
    找出「其實是規模／流動性代理」的因子（相關度那一道）。

    ⚠️ 刻意不用前置專案那種寫死的名單（`["liq_amt", "neg_size"]`）——
    那只認得 DFS 因子的名字，agent 自己挖的因子若是規模的變體，名單抓不到。
    改成量測每個因子與 log(成交金額) 的**橫斷面排名相關**（逐月算再平均），
    這是行為上的判定，對兩邊的因子一視同仁。
    """
    mb = pd.read_parquet(ROOT / fl.CFG["paths"]["monthly_base"],
                         columns=["stock_id", "ym", "amt_21"])
    mb["ym"] = mb["ym"].astype(str)
    mb["stock_id"] = mb["stock_id"].astype(str)
    d = proc.merge(mb, on=["stock_id", "ym"], how="left")
    d["size"] = np.log1p(d["amt_21"])
    d = d.dropna(subset=["size"])

    out = {}
    for f in feats:
        per_month = []
        for ym, g in d.groupby("ym"):
            s = g[[f, "size"]].dropna()
            if len(s) < bt.MIN_STOCKS or s[f].nunique() < 2 or s["size"].nunique() < 2:
                continue
            rho = s[f].rank().corr(s["size"].rank())
            if pd.notna(rho):
                per_month.append(rho)
        out[f] = float(np.mean(per_month)) if per_month else np.nan
    corr = pd.Series(out)
    return corr, [f for f in feats if abs(corr.get(f, 0)) > threshold]


def cmd_importance(which: str, kind: str, span: str, factors, names_map=None):
    """LightGBM 特徵重要性 + SHAP，並跑一次「排除規模代理」的中性版。"""
    if kind != "lgbm":
        print(f"（--importance 需要 lgbm；已自動改用 lgbm，忽略 --model {kind}）")
        kind = "lgbm"
    d, feats, names, proc = build_scores(which, kind, factors)

    from memory import Memory
    lib = Memory().library()
    by_formula = formula_size_proxies(lib, feats)
    corr, by_corr = detect_size_proxies(proc, feats)
    size_proxies = sorted(set(by_formula) | set(by_corr))

    print(f"\n=== 規模代理偵測 ===")
    print("兩道判準：(1) 公式直接用到 " + "／".join(SIZE_FIELDS) +
          f"　(2) 與 log(成交金額) 的橫斷面排名相關 |ρ| > {SIZE_CORR_THRESHOLD}")
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
    for f, v in top.items():
        why = []
        if f in by_formula:
            why.append("公式")
        if f in by_corr:
            why.append("相關度")
        flag = f"  ← 規模代理（{'＋'.join(why)}）" if why else ""
        print(f"  {f:<8}{(names.get(f, '') or '')[:16]:<18}{v:>+7.3f}{flag}")
    only_formula = [f for f in by_formula if f not in by_corr]
    if only_formula:
        print(f"\n⚠️ 這些因子的公式裡有 {'／'.join(SIZE_FIELDS)}，但量測相關度未達門檻："
              f"{only_formula}")
        for f in only_formula:
            print(f"     {f} {names.get(f, '')}：{(lib.get(f) or {}).get('formula')}")
        print("   公式是定義證據，比相關度更硬——一併列為規模代理。")
    print(f"\n判定為規模代理的 {len(size_proxies)} 個：{size_proxies or '（無）'}")

    # 重要性：在訓練期末尾擬合一次（不用來預測，只看模型學到什麼）
    import lightgbm as lgb
    months = sorted(proc["ym"].unique())
    cut = months[min(fl.MIN_TRAIN_MONTHS, len(months) - 1)]
    tr = proc[proc["ym"] <= cut]
    print(f"\n（重要性用 ym ≤ {cut} 的訓練資料擬合一次，{len(tr):,} 列；"
          f"不參與任何預測，純粹用來看模型倚重哪些因子）")
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              max_depth=5, min_child_samples=200, subsample=0.8,
                              colsample_bytree=0.6, reg_lambda=5.0,
                              random_state=0, n_jobs=-1, verbose=-1)
    model.fit(tr[feats].values, tr["y"].values)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(
        ascending=False)
    print("\n=== LightGBM 特徵重要性（前 15）===")
    for f, v in imp.head(15).items():
        print(f"  {f:<8}{(names.get(f, '') or '')[:18]:<20}{v:>6}")

    try:
        import shap
        samp = tr[feats].sample(min(2000, len(tr)), random_state=0)
        sv = np.abs(shap.TreeExplainer(model).shap_values(samp.values)).mean(0)
        s = pd.Series(sv, index=feats).sort_values(ascending=False)
        print("\n=== SHAP 平均絕對貢獻（前 15）===")
        for f, v in s.head(15).items():
            print(f"  {f:<8}{(names.get(f, '') or '')[:18]:<20}{v:>10.5f}")
    except ImportError:
        print("\n（未安裝 shap，略過；pip install shap 可開啟）")

    # 規模中性版
    if not size_proxies:
        print(f"\n（沒有因子被判定為規模代理，不需要跑中性版——這本身就是好消息）")
        return

    neutral = [f for f in feats if f not in size_proxies]
    print(f"\n=== 規模中性版：排除 {len(size_proxies)} 個規模代理，"
          f"剩 {len(neutral)} 個因子 ===")
    a = fl.score_and_backtest(proc, feats, kind, span)
    b = fl.score_and_backtest(proc, neutral, kind, span)
    print(f"{'版本':<12}{'因子數':>6}{'CAGR':>9}{'超額':>9}{'IR':>7}{'換手':>7}")
    for lab, pf, n in (("全部因子", a, len(feats)),
                       ("規模中性", b, len(neutral))):
        if pf:
            print(f"{lab:<12}{n:>6}{pf['CAGR']:>9.2%}{pf['Excess']:>9.2%}"
                  f"{pf['IR']:>7.2f}{pf['Turnover']:>7.1%}")
    if not (a and b):
        return
    drop = (a["IR"] - b["IR"]) / a["IR"]

    # ── 對照組：拿掉「同樣重要、但不是規模代理」的因子，掉幅是多少？ ──
    # 沒有這個對照，中性版的掉幅無法解讀：拿掉任何一個重要因子 IR 都會掉，
    # 問題是「掉得比一般情況多嗎」。用 SHAP 排名相近的非規模因子當對照。
    ctrl_pool = [f for f in imp.index if f not in size_proxies][:len(size_proxies) * 4]
    ctrl = []
    for f in ctrl_pool[:6]:
        pf = fl.score_and_backtest(proc, [x for x in feats if x != f], kind, span)
        if pf:
            ctrl.append(((a["IR"] - pf["IR"]) / a["IR"], f))
    if ctrl:
        med = float(np.median([c[0] for c in ctrl]))
        print(f"\n對照組（逐一拿掉重要性前段、但非規模代理的因子，各 1 個）：")
        for v, f in sorted(ctrl, reverse=True):
            print(f"   拿掉 {f:<8}{(names.get(f, '') or '')[:16]:<18}IR 掉幅 {v:>+7.1%}")
        print(f"   中位掉幅 {med:+.1%}")
        print(f"\n判讀：規模中性版掉了 {drop:+.1%}，對照組單一因子中位掉幅 {med:+.1%}。")
        if drop > max(med * 2, 0.15):
            print(f"   → 掉幅明顯高於對照組，**edge 確實有相當部分來自規模／流動性傾斜**。")
        else:
            print(f"   → 掉幅與對照組同量級，規模傾斜不是主要來源。")
    else:
        print(f"\n判讀：中性版掉幅 {drop:+.1%}；掉幅 < 20% 通常代表 edge 不只是小型股傾斜。")


# ---------------------------------------------------------------------------
# 4. Deflated Sharpe + PBO
# ---------------------------------------------------------------------------

def psr(sr, sr_star, T, skew, kurt):
    """Probabilistic Sharpe：P(真實 Sharpe > sr_star)。sr 為每期(月) Sharpe。"""
    from scipy.stats import norm
    den = np.sqrt(1 - skew * sr + ((kurt - 1) / 4.0) * sr ** 2)
    return float(norm.cdf((sr - sr_star) * np.sqrt(T - 1) / den))


def deflated_sharpe(rets: pd.Series, trial_sharpes):
    """
    Bailey & López de Prado 的 Deflated Sharpe。

    核心觀念：一個回測 Sharpe 之所以好看，可能只是「試了 N 個策略挑出的幸運兒」。
    DSR 依試驗數 N、各試驗 Sharpe 的變異、樣本長度、報酬的偏態與峰態，
    算出「期望中的最大 Sharpe」當門檻，再問真實 Sharpe 超過門檻的機率。
    """
    from scipy.stats import norm
    r = pd.Series(rets).dropna()
    T = len(r)
    sr = r.mean() / r.std()
    skew, kurt = r.skew(), r.kurtosis() + 3.0   # pandas 給超額峰態，轉回非超額
    ts = np.array([s for s in trial_sharpes if np.isfinite(s)])
    N = len(ts)
    if N < 2:
        raise SystemExit("試驗數不足，無法估去膨脹門檻")
    g = 0.5772156649                            # Euler–Mascheroni
    sr0 = np.sqrt(ts.var(ddof=1)) * (
        (1 - g) * norm.ppf(1 - 1.0 / N) + g * norm.ppf(1 - 1.0 / (N * np.e)))
    return psr(sr, sr0, T, skew, kurt), sr, sr0, N, T


def cscv_pbo(R: pd.DataFrame, S: int = 12):
    """
    CSCV 估 Probability of Backtest Overfitting。

    把時間切成 S 塊、窮舉一半當樣本內、另一半當樣本外，看「樣本內最好的策略
    到樣本外是否落在中位數以下」的比例。PBO < 0.5 代表選擇流程沒有嚴重過擬合。
    """
    R = R.dropna(axis=0, how="any")
    T, N = R.shape
    if T < S * 2 or N < 3:
        raise SystemExit(f"樣本不足做 CSCV（月數 {T}、策略數 {N}）")
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for isin in combinations(range(S), S // 2):
        is_rows = np.concatenate([blocks[b] for b in isin])
        oos_rows = np.concatenate([blocks[b] for b in range(S) if b not in isin])
        Ris, Roos = R.iloc[is_rows], R.iloc[oos_rows]
        sd_is = Ris.std()
        best = (Ris.mean() / sd_is.replace(0, np.nan)).idxmax()
        sd_oos = Roos.std()
        rank = (Roos.mean() / sd_oos.replace(0, np.nan)).rank().loc[best]
        omega = rank / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return float((logits <= 0).mean()), logits


# R10：單因子方向的鎖定點。因子當初就是用 sub_train + validation 選出來的，
# 方向也必須在同一個窗內決定——之後的任何期間都不得回頭改變它。
ORIENT_CUTOFF = fl.SPANS["validation"][1]


def _active_returns(d: pd.DataFrame, col: str, span: str, orient: bool):
    """
    單一策略的樣本外「主動報酬」= 產業內前 top_q 等權 − 全池等權基準。

    ⚠️ 一定要減掉基準。DSR / PBO 測的是「相對選股技巧」；不減的話，
    所有策略都吃同一份多頭 beta，試驗集的 Sharpe 變異被壓縮，
    去膨脹門檻會被低估，DSR 就會虛高。
    """
    sub = d[["stock_id", "group", "ym", "fwd_ret_1m", col]].dropna(subset=[col])
    if len(sub) < 100:
        return None
    sub = sub.rename(columns={col: "score"})
    if orient:
        # 單因子策略要決定方向，否則負向因子全被判無效。
        # ⛔ R10：方向**只能用選取窗（≤ validation 末端）**決定，不能用全期。
        #    用全期 IC 定向再切 test，等於讓 test 期的表現回頭定義策略本身：
        #    一個在 train 為負、在 test 強正的因子會被事後翻向，然後在 test
        #    上量出漂亮的 Sharpe——那個 Sharpe 是自己造出來的。
        ic = _mean_ic(sub[sub["ym"] <= ORIENT_CUTOFF])
        if pd.isna(ic):
            return None
        if ic < 0:
            sub = sub.assign(score=-sub["score"])
    rets = bt.slice_span(bt.portfolio_returns(sub), fl.SPANS.get(span))
    if len(rets) < 12:
        return None
    return bt.excess_vs_benchmark(rets)


def _mean_ic(sub: pd.DataFrame) -> float:
    """產業內逐月 Spearman IC 的平均（口徑與 eval_candidates 一致）。"""
    ics = []
    for ym, g in sub.groupby("ym"):
        per = []
        for grp, gg in g.groupby("group"):
            s = gg[["score", "fwd_ret_1m"]].dropna()
            if len(s) < bt.MIN_STOCKS:
                continue
            if s["score"].nunique() < 2 or s["fwd_ret_1m"].nunique() < 2:
                continue
            v = s["score"].rank().corr(s["fwd_ret_1m"].rank())
            if pd.notna(v):
                per.append(v)
        if per:
            ics.append(np.mean(per))
    return float(np.mean(ics)) if ics else np.nan


def _print_trial_universe_scope(n_factors: int, n_models: int) -> None:
    """
    R11：DSR／PBO 涵蓋的是**哪一個候選宇宙**，必須跟著結果一起講。

    這不是公式寫錯的問題，是**統計結論的適用範圍**問題。對「存活下來的因子」
    算 DSR，證明的是「在這 N 條序列裡挑到最好的那條，不是純運氣」；
    它**不等於**「整個挖礦流程已經去偏」。
    """
    import json as _json
    try:
        att = len(list((ROOT / "memory" / "attempts").glob("*.json")))
    except Exception:
        att = None
    print("\n⚠️ 這個 DSR／PBO 涵蓋的候選宇宙（R11）")
    print(f"   納入：{n_factors} 個**已入庫**因子各自的單因子策略 + {n_models} 個合成模型")
    print("   未納入：")
    if att:
        print(f"     · {att} 筆 attempts 裡被漏斗刷掉的候選"
              f"（入庫率僅約 {26 / att * 100:.0f}%，搜尋量遠大於試驗池）")
    print("     · 參考因子池的選取與去重過程")
    print("     · 產業範圍核准、模型與超參數的選擇")
    print("   → 可以說的是「在這個試驗池內，最佳策略不是多重測試的產物」；")
    print("     **不能**說「整個挖礦流程已經去偏」。後者需要獨立 holdout 或巢狀評估。")
    print("   ⚠️ 也不能草率把試驗數改成 attempts 總數——那些候選高度相關，"
          "\n      獨立性假設不成立，只會把門檻灌高成另一種假象。")


def cmd_dsr(which: str, kind: str, span: str, factors, blocks: int = 12):
    """
    用「每個因子各自的策略 + 每個模型」當試驗母體，估 DSR 與 PBO。

    試驗母體要誠實地代表「我們搜尋過的空間」：51 個入庫因子各自的
    產業內前 10% 策略，加上 equal / ridge / lgbm 三個合成模型。
    """
    df, feats, names = fl.load_panel(which, factors)
    proc = fl.prep(df, feats)
    models = ("equal", "ridge", kind) if kind not in ("equal", "ridge") \
        else ("equal", "ridge")
    models = tuple(dict.fromkeys(models))

    print(f"建立試驗母體：{len(feats)} 個單因子策略 + {len(models)} 個模型"
          f"（{span} 期，需數分鐘）…")
    cols = {}
    for f in feats:
        base = proc[["stock_id", "group", "ym", "fwd_ret_1m"]].copy()
        base[f] = proc[f]
        r = _active_returns(base, f, span, orient=True)
        if r is not None:
            cols[f] = r
    for m in models:
        base = proc[["stock_id", "group", "ym", "fwd_ret_1m"]].copy()
        base[m] = fl.walk_forward(proc, feats, m)
        r = _active_returns(base, m, span, orient=False)
        if r is not None:
            cols[m] = r
    R = pd.DataFrame(cols).sort_index()
    target = kind if kind in R.columns else R.columns[-1]

    trial_sr = [R[c].dropna().mean() / R[c].dropna().std()
                for c in R.columns if R[c].dropna().std() > 0]
    dsr, sr, sr0, N, T = deflated_sharpe(R[target], trial_sr)

    print(f"\n===== Deflated Sharpe（{target} 的『主動報酬』，已剝離市場 beta）=====")
    print(f"樣本外月數 T：{T}　試驗數 N：{N}")
    print(f"主動報酬每月 Sharpe（≈月 IR）：{sr:.3f}"
          f"（年化 {sr * np.sqrt(ANN):.2f}）")
    print(f"去膨脹門檻 SR0（N 次試驗的期望最大值）：{sr0:.3f}"
          f"（年化 {sr0 * np.sqrt(ANN):.2f}）")
    print(f"**Deflated Sharpe：{dsr:.3f}**")
    print("判讀：> 0.95 表示扣掉「試了 N 個策略」的多重測試後，選股超額仍然顯著。")
    _print_trial_universe_scope(len(feats), len(models))

    try:
        pbo, _ = cscv_pbo(R, S=blocks)
    except SystemExit as e:
        print(f"\n（PBO 略過：{e}）")
        return {"DSR": dsr, "SR": sr, "SR0": sr0, "N": N, "T": T}
    print(f"\n===== PBO（回測過擬合機率，CSCV）=====")
    print(f"策略母體 N={R.shape[1]}　切塊 S={blocks}"
          f"（C({blocks},{blocks//2}) = {len(list(combinations(range(blocks), blocks//2)))} 組）")
    print(f"**PBO：{pbo:.3f}**")
    print("判讀：< 0.5 代表「樣本內最好的策略」到樣本外沒有系統性退步，"
          "\n     選擇流程沒有嚴重過擬合；越接近 0 越好。")
    n_models = sum(1 for m in models if m in R.columns)
    if pbo < 0.05 and n_models < R.shape[1] * 0.2:
        print(f"\n⚠️ PBO 極低（{pbo:.3f}）要打個折扣看。試驗母體裡 "
              f"{R.shape[1] - n_models} 個是單因子策略、只有 {n_models} 個是合成模型，"
              f"\n   而合成模型在任何切塊都穩定勝過單因子——樣本內最佳幾乎總是模型，"
              f"\n   樣本外也是，PBO 於是趨近 0。PBO 最有鑑別力的用法是**互相可比的**"
              f"\n   策略母體（例如同一個模型的多組超參數）。這裡的數字說明的是"
              f"\n   「合成勝過單因子」很穩健，而不是「這組超參數沒過擬合」。")
    return {"DSR": dsr, "PBO": pbo, "SR": sr, "SR0": sr0, "N": N, "T": T}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="合成分數的過擬合／穩健性診斷")
    ap.add_argument("--all", action="store_true", help="四項全跑")
    ap.add_argument("--regime", action="store_true", help="多空 régime 分解")
    ap.add_argument("--breadth", action="store_true",
                    help="逐月超額分布、集中度、持股產業組成")
    ap.add_argument("--timing", action="store_true",
                    help="擇時研究：上限與損益兩平準確率")
    ap.add_argument("--regime-factors", action="store_true",
                    help="逐因子的多空月 ICIR 差異（狀態模型該接的地方）")
    ap.add_argument("--liquidity", action="store_true", help="流動性歸因 + 壓力測試")
    ap.add_argument("--importance", action="store_true",
                    help="特徵重要性 + SHAP + 規模中性")
    ap.add_argument("--dsr", action="store_true", help="Deflated Sharpe + PBO")
    ap.add_argument("--set", default="all", choices=["own", "reference", "all"])
    ap.add_argument("--factors", help="逗號分隔的 factor_id，覆寫 --set")
    ap.add_argument("--model", default="lgbm",
                    choices=["equal", "ridge", "lgbm"])
    ap.add_argument("--span", default="test",
                    choices=["sub_train", "validation", "test"])
    ap.add_argument("--blocks", type=int, default=12, help="CSCV 的切塊數")
    ap.add_argument("--save", help="把可量化的結果存成 JSON")
    a = ap.parse_args()

    if a.span != "test":
        print(f"⚠️ {a.span} 期對 agent 的因子而言是 in-sample，數字會虛高。\n")
    factors = [f.strip() for f in a.factors.split(",")] if a.factors else None
    picked = (a.regime, a.breadth, a.timing, a.regime_factors,
              a.liquidity, a.importance, a.dsr)
    run_all = a.all or not any(picked)
    out = {}

    if run_all or a.regime:
        cmd_regime(a.set, a.model, a.span, factors)
    if run_all or a.breadth:
        cmd_breadth(a.set, a.model, a.span, factors)
    if run_all or a.regime_factors:
        cmd_regime_factors(a.set, a.span, factors)
    if a.timing:            # --all 不含（模擬要跑幾千次回測，很慢）
        out["timing"] = cmd_timing(a.set, a.model, a.span, factors)
    if run_all or a.liquidity:
        out["liquidity"] = cmd_liquidity(a.set, a.model, a.span, factors)
    if run_all or a.importance:
        cmd_importance(a.set, a.model, a.span, factors)
    if run_all or a.dsr:
        out["dsr"] = cmd_dsr(a.set, a.model, a.span, factors, a.blocks)

    if a.save and out:
        Path(a.save).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                           default=float), encoding="utf-8")
        print(f"\n已存 {a.save}")


if __name__ == "__main__":
    main()
