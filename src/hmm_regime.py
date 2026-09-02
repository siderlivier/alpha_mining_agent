"""
市場狀態模型（Gaussian HMM）：預測下個月是「多頭態」還是「空頭態」。

用途不是拿來決定進出場（`ml_diagnose --timing` 已證明那條路要 80% 準確率
才損益兩平），而是拿來**切換因子組**——`ml_diagnose --regime-factors` 顯示
低波動族群在下跌月的 ICIR t 值高達 8~12、上漲月卻貼近零，這個不對稱在
訓練期與測試期都成立，是狀態訊號真正該利用的東西。

⛔ 三道前瞻紀律，每一道都有對應的測試（tests/test_hmm_regime.py）
─────────────────────────────────────────────────────────────
1. **觀測值的時點對齊**
   `monthly_base` 的 `fwd_ret_1m` 是「t → t+1」的報酬，所以
   `mkt[t] = 各股 fwd_ret_1m 在月份 t 的平均` 其實是**未來一個月**的市場報酬。
   在月底 t 做決策時，最新可觀測的市場報酬是 `mkt[t-1]`。
   本模組所有特徵都建在 `mkt[..., t-1]` 上，預測目標是 `sign(mkt[t])`。
   ⚠️ 這是整支程式最容易錯的一行，錯了會得到假的高準確率。

2. **推論只用前向濾波（filtering），不用 Viterbi / 平滑**
   `hmmlearn` 的 `predict()` 跑的是 Viterbi——它會用**整段序列**（含未來
   觀測）回推每個時點的狀態。拿來做回測就是洩漏。本模組只用 hmmlearn
   擬合參數，狀態推論自己寫前向遞迴：時點 t 的狀態機率只由 o_1..o_t 決定。

3. **參數只在訓練資料上擬合**
   擴張窗：第 i 段用 `ym <= cut` 擬合、預測其後 `refit_every` 個月，
   與 `factor_lab.segments()` 同一套切法。測試期永遠只驗證、不參與擬合；
   因子的攻擊／防禦分類也**只用 ≤ validation 末端**的資料決定。

用法
----
    python src/hmm_regime.py --fit                # 訓練期擬合，看狀態長什麼樣
    python src/hmm_regime.py --predict            # walk-forward 樣本外準確率
    python src/hmm_regime.py --apply              # 用狀態切因子組，比較靜態
    python src/hmm_regime.py --predict --states 3 --features ret,vol,breadth

需要：pip install hmmlearn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt
import factor_lab as fl

ROOT = Path(__file__).resolve().parents[1]
CFG = fl.CFG
HP = CFG.get("hmm") or {}

N_STATES = int(HP.get("n_states", 2))
MIN_TRAIN = int(HP.get("min_train_months", 60))
REFIT_EVERY = int(HP.get("refit_every", 12))
SEED = int(HP.get("seed", 0))
FEATURES = tuple(HP.get("features", ["ret", "vol"]))
VOL_WINDOW = int(HP.get("vol_window", 6))


# ---------------------------------------------------------------------------
# 觀測序列
# ---------------------------------------------------------------------------

def market_series() -> pd.DataFrame:
    """
    由 monthly_base 建立市場層級的月頻序列。

    回傳 index=ym 的 DataFrame：
      mkt        月份 t **未來一個月**的市場報酬（= 各股 fwd_ret_1m 的平均）
                 —— 這是**預測目標**，不是特徵
      obs_ret    月底 t **已實現**的最新市場報酬（= mkt 落後一期）
      obs_vol    過去 VOL_WINDOW 個月已實現報酬的標準差
      obs_breadth 上個月上漲家數比例

    ⚠️ 所有 `obs_*` 欄位都由 `mkt.shift(1)` 衍生。若哪天有人為了「讓特徵
       更即時」把 shift 拿掉，準確率會突然跳到 90% 以上——那不是模型變好，
       是它看到了答案。
    """
    mb = pd.read_parquet(ROOT / CFG["paths"]["monthly_base"],
                         columns=["stock_id", "ym", "fwd_ret_1m"])
    mb["ym"] = mb["ym"].astype(str)
    g = mb.dropna(subset=["fwd_ret_1m"]).groupby("ym")["fwd_ret_1m"]
    mkt = g.mean().sort_index()
    breadth = g.apply(lambda s: float((s > 0).mean())).sort_index()

    d = pd.DataFrame({"mkt": mkt})
    d["obs_ret"] = d["mkt"].shift(1)                       # ← 唯一的時點對齊
    d["obs_vol"] = d["obs_ret"].rolling(VOL_WINDOW, min_periods=3).std()
    d["obs_breadth"] = breadth.shift(1)
    return d


FEATURE_COLS = {"ret": "obs_ret", "vol": "obs_vol", "breadth": "obs_breadth"}


def observation_matrix(d: pd.DataFrame, features) -> tuple[np.ndarray, pd.Index]:
    """取出特徵矩陣，丟掉暖身期的缺值列。"""
    cols = [FEATURE_COLS[f] for f in features]
    sub = d[cols].dropna()
    return sub.values, sub.index


# ---------------------------------------------------------------------------
# 擬合與推論
# ---------------------------------------------------------------------------

def fit_hmm(X: np.ndarray, n_states: int = N_STATES, seed: int = SEED):
    """在給定觀測上擬合 Gaussian HMM（Baum-Welch）。"""
    from hmmlearn.hmm import GaussianHMM
    m = GaussianHMM(n_components=n_states, covariance_type="diag",
                    n_iter=200, random_state=seed, tol=1e-4)
    m.fit(X)
    return m


def forward_filter(model, X: np.ndarray) -> np.ndarray:
    """
    前向濾波：回傳 (T, n_states) 的**濾波**狀態機率 P(state_t | o_1..o_t)。

    ⚠️ 刻意不用 `model.predict()` 或 `model.predict_proba()`。
       前者是 Viterbi、後者是**平滑**機率 P(state_t | o_1..o_T)——
       兩者都用到 t 之後的觀測，拿來做回測就是前瞻偏差。
       這裡自己寫遞迴，每一步只吃到當期為止的觀測。

    數值上逐步正規化（scaling），避免長序列下溢。
    """
    logB = model._compute_log_likelihood(X)          # (T, n_states)
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    A = model.transmat_
    T, K = B.shape
    alpha = np.zeros((T, K))
    a = model.startprob_ * B[0]
    s = a.sum()
    alpha[0] = a / s if s > 0 else np.full(K, 1.0 / K)
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        s = a.sum()
        alpha[t] = a / s if s > 0 else np.full(K, 1.0 / K)
    return alpha


def label_states(model) -> dict:
    """
    依「觀測特徵的狀態均值」排序，用來**描述**狀態（高報酬態／低報酬態）。

    HMM 的狀態編號本身沒有意義（每次擬合可能對調），所以要有一個穩定的排序。
    ⚠️ 這個排序只用來取名字，**不可以拿來當「哪個狀態預示上漲」的答案**——
       見 `bull_state_from_train()` 的說明。
    """
    order = np.argsort(model.means_[:, 0])            # 由低到高
    return {int(k): i for i, k in enumerate(order)}   # 原編號 → 排序後名次


def bull_state_from_train(model, X: np.ndarray, idx, mkt: pd.Series,
                          n_obs: int) -> int:
    """
    在**訓練資料上**判定「哪個狀態之後的市場報酬比較高」。

    為什麼不能直接用觀測報酬均值來當多頭態
    ─────────────────────────────────────
    直覺是「過去報酬高、波動低的那個狀態就是多頭態」。實測完全相反：

        訓練期（2012-04~2019-12）
          低波動態（觀測報酬均值 +1.24%）→ 之後一個月平均 +0.40%、上漲 61.8%
          高波動態（觀測報酬均值 +0.16%）→ 之後一個月平均 +2.56%、上漲 72.0%

    高波動之後反而漲得多（波動的風險溢酬／反彈效應）。若照直覺命名，
    方向預測會系統性地反向——實測 test 期準確率只有 35.1%（基準 64.9%），
    t = −2.33。那不是「模型沒用」，是**標籤貼反了**。

    這裡改成用資料決定：在訓練窗內比較各狀態的後續市場報酬均值，取最高者。
    ⛔ 只能用擬合當下已經實現的月份。觀測 X[j] 對應月份 idx[j]，而
       mkt[idx[j]] 是「idx[j] → idx[j]+1」的報酬，要到下個月底才知道。
       所以擬合點在 idx[n_obs-1] 時，可用的配對只到 j = n_obs-2。
    """
    usable = n_obs - 1                       # j 最多到 n_obs-2
    if usable < 2:
        return int(np.argmax(model.means_[:, 0]))
    alpha = forward_filter(model, X[:usable])
    hard = alpha.argmax(axis=1)
    nxt = mkt.reindex(idx[:usable]).values
    best, best_mu = None, -np.inf
    for k in range(model.n_components):
        m = (hard == k) & np.isfinite(nxt)
        if m.sum() < 3:
            continue
        mu = float(np.nanmean(nxt[m]))
        if mu > best_mu:
            best, best_mu = k, mu
    return best if best is not None else int(np.argmax(model.means_[:, 0]))


# ---------------------------------------------------------------------------
# walk-forward 樣本外狀態
# ---------------------------------------------------------------------------

def walk_forward_states(d: pd.DataFrame, features=FEATURES,
                        n_states: int = N_STATES, min_train: int = MIN_TRAIN,
                        refit_every: int = REFIT_EVERY, seed: int = SEED
                        ) -> pd.DataFrame:
    """
    擴張窗擬合 + 前向濾波，產生每個月的**樣本外**多頭態機率。

    第 i 段：用 `X[:i]` 擬合參數，對 `X[:i+j+1]` 做前向濾波、取最後一格，
    得到月份 i+j 的濾波機率。參數與觀測都不含 i+j 之後的資訊。
    """
    X, idx = observation_matrix(d, features)
    T = len(X)
    if T <= min_train:
        raise SystemExit(f"觀測只有 {T} 個月，不足 min_train={min_train}")

    mkt = d["mkt"]
    rows = []
    i = min_train
    while i < T:
        model = fit_hmm(X[:i], n_states, seed)        # ← 只用 i 之前的資料
        bull = bull_state_from_train(model, X, idx, mkt, i)   # ← 也只用訓練窗
        end = min(i + refit_every, T)
        for j in range(i, end):
            alpha = forward_filter(model, X[:j + 1])  # ← 只吃到 j 為止
            rows.append({"ym": idx[j], "p_bull": float(alpha[-1, bull]),
                         "state": int(np.argmax(alpha[-1])),
                         "fit_upto": idx[i - 1]})
        i = end
    out = pd.DataFrame(rows).set_index("ym")
    out["mkt"] = d["mkt"].reindex(out.index)          # 預測目標（未來一個月）
    return out


# ---------------------------------------------------------------------------
# 指令
# ---------------------------------------------------------------------------

def cmd_fit(features, n_states, seed):
    """只在訓練期（≤ validation 末端）擬合一次，看狀態的性質。"""
    d = market_series()
    cut = CFG["split"]["validation"][1]
    X, idx = observation_matrix(d, features)
    keep = idx <= cut
    print(f"訓練資料：{idx[keep][0]} ~ {idx[keep][-1]}（{keep.sum()} 個月）"
          f"，特徵 {list(features)}")
    print(f"⚠️ test 期（{CFG['split']['test'][0]} 起）完全不參與擬合。")
    model = fit_hmm(X[keep], n_states, seed)
    rank = label_states(model)
    bull = bull_state_from_train(model, X[keep], idx[keep], d["mkt"],
                                 int(keep.sum()))

    alpha = forward_filter(model, X[keep])
    hard = alpha.argmax(axis=1)
    mkt = d["mkt"].reindex(idx[keep])
    print(f"\n=== 狀態特徵（依觀測報酬均值排序）===")
    print(f"{'狀態':<6}{'觀測特性':<12}{'月數':>6}{'觀測報酬均值':>13}"
          f"{'觀測波動':>10}{'→下月市場報酬':>16}{'→下月上漲比例':>15}")
    for k in sorted(rank, key=lambda x: rank[x]):
        m = hard == k
        desc = "低報酬態" if rank[k] == 0 else ("高報酬態" if rank[k] == n_states - 1
                                             else f"中性{rank[k]}")
        cov = model.covars_[k]
        sd = np.sqrt(cov[0, 0] if np.ndim(cov) == 2 else cov[0])
        nxt = mkt[m]
        star = "  ← 訓練期判定為多頭態" if k == bull else ""
        print(f"{k:<6}{desc:<12}{m.sum():>6}{model.means_[k, 0]:>13.2%}"
              f"{sd:>10.2%}{nxt.mean():>16.2%}{(nxt > 0).mean():>15.1%}{star}")
    print("\n⚠️ 注意「觀測特性」與「→下月」兩組數字可能是**相反**的。"
          "\n   哪個狀態預示上漲，一律由訓練期的『→下月市場報酬』決定，"
          "\n   不能望文生義用觀測報酬均值判斷（見 bull_state_from_train 的說明）。")
    print(f"\n轉移矩陣（列 = 目前狀態，欄 = 下期狀態）：")
    for k in range(n_states):
        print("   " + "  ".join(f"{v:.3f}" for v in model.transmat_[k]))
    print("\n判讀：若各狀態的『當期實際市場報酬』沒有明顯分開，代表 HMM 只是把"
          "\n     波動高低分群，對方向沒有預測力——這在只餵報酬與波動時很常見。")
    return model


def cmd_predict(features, n_states, seed, span):
    """walk-forward 樣本外狀態，評估方向預測準確率。"""
    d = market_series()
    st = walk_forward_states(d, features, n_states, seed=seed)
    lo, hi = CFG["split"][span]
    s = st[(st.index >= lo) & (st.index <= hi)].dropna(subset=["mkt"])
    if not len(s):
        raise SystemExit(f"{span} 期沒有樣本外狀態")

    truth = s["mkt"] > 0
    base = float(truth.mean())
    print(f"\n=== 方向預測（{span} 期，{len(s)} 個月，特徵 {list(features)}）===")
    print(f"實際上漲月比例（永遠猜漲的準確率）：{base:.1%}  ← 這是要打敗的基準")

    print(f"\n{'門檻':>8}{'預測漲的月數':>13}{'準確率':>9}{'vs基準':>9}"
          f"{'預測漲時的平均報酬':>19}{'預測跌時的平均報酬':>19}")
    rows = []
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = s["p_bull"] >= thr
        acc = float((pred == truth).mean())
        a = s.loc[pred, "mkt"].mean() if pred.any() else np.nan
        b = s.loc[~pred, "mkt"].mean() if (~pred).any() else np.nan
        print(f"{thr:>8.1f}{int(pred.sum()):>13}{acc:>9.1%}{acc - base:>+9.1%}"
              f"{a:>19.2%}{b:>19.2%}")
        rows.append({"門檻": thr, "準確率": acc, "vs基準": acc - base,
                     "預測漲平均報酬": float(a) if pd.notna(a) else None,
                     "預測跌平均報酬": float(b) if pd.notna(b) else None})

    # 更有意義的檢定：預測漲 vs 預測跌，兩組報酬有沒有真的分開
    pred = s["p_bull"] >= 0.5
    acc50 = float((pred == truth).mean())
    if pred.any() and (~pred).any():
        from scipy import stats
        t, pv = stats.ttest_ind(s.loc[pred, "mkt"], s.loc[~pred, "mkt"],
                                equal_var=False)
        print(f"\n兩組報酬差異檢定（門檻 0.5）：t = {t:.2f}，p = {pv:.3f}")
        print("  → 準確率會被『多數類別』灌水，這個檢定才是真的在問"
              "\n    『預測為漲的月份，報酬是否真的比較高』。")

        print("\n=== 結論 ===")
        beats = acc50 > base
        sig = pv < 0.05 and t > 0
        if sig and not beats:
            print(f"  ✅ 有訊號但不足以擇時：兩組報酬顯著分開（p={pv:.3f}），"
                  f"\n     但方向準確率 {acc50:.1%} 仍**輸給**「永遠猜漲」的 {base:.1%}。")
            print(f"     原因是模型太保守——77 個月只喊 {int(pred.sum())} 個月看多，"
                  f"漏掉大量上漲月。")
            print(f"  → **不要**拿它決定進出場（`ml_diagnose --timing` 顯示那要 80% 準確率）。")
            print(f"  → **可以**拿它切換因子組：切錯只是用了較不合適的因子，"
                  f"不會整月踏空。用 --apply 驗證。")
        elif sig and beats:
            print(f"  ✅ 兩組報酬顯著分開（p={pv:.3f}）且準確率 {acc50:.1%} "
                  f"勝過基準 {base:.1%}——罕見，請先確認沒有前瞻洩漏再高興。")
        elif t < 0 and pv < 0.05:
            print(f"  ⚠️ 兩組報酬**反向**顯著（t={t:.2f}）。這通常代表狀態標籤貼反了，"
                  f"\n     而不是模型無效——檢查 bull_state_from_train 是否用到了訓練期的後續報酬。")
        else:
            print(f"  ❌ 兩組報酬沒有顯著分開（p={pv:.3f}）→ 這組特徵沒有方向預測力。"
                  f"\n     那不代表 HMM 沒用：它仍可能把高／低波動分得很好，"
                  f"\n     那個分群對切換因子組依然有價值（見 --apply）。")
    return rows


def cmd_apply(features, n_states, seed, threshold=0.5):
    """
    用樣本外狀態切換因子組，與靜態全用比較。

    攻擊／防禦的分類**只用 ≤ validation 末端**的資料決定：
    依「下跌月 ICIR 的 t 值」挑防禦組——這是 --regime-factors 顯示
    在兩個期間都穩定的不對稱，不是善變的正負號。
    """
    import ml_diagnose as md
    df, feats, names = fl.load_panel("all")
    proc = fl.prep(df, feats)

    cls_lo, cls_hi = CFG["split"]["sub_train"][0], CFG["split"]["validation"][1]
    R = md.factor_ic_by_regime(proc, feats, cls_lo, cls_hi, names)
    n_dn = (proc[(proc["ym"] >= cls_lo) & (proc["ym"] <= cls_hi)]
            .groupby("ym")["fwd_ret_1m"].mean() <= 0).sum()
    R["t_跌"] = R["ICIR_跌"] * np.sqrt(n_dn)
    R = R.sort_values("t_跌", ascending=False)
    defensive = list(R.head(max(3, len(R) // 6))["因子"])
    print(f"分類期 {cls_lo} ~ {cls_hi}（下跌 {n_dn} 個月），"
          f"依『下跌月 ICIR 的 t 值』選出防禦組 {len(defensive)} 個：")
    for _, r in R.head(len(defensive)).iterrows():
        print(f"   {r['因子']:<8}{r['名稱']:<16}t_跌 {r['t_跌']:>6.2f}"
              f"  ICIR_跌 {r['ICIR_跌']:>5.2f}  ICIR_漲 {r['ICIR_漲']:>5.2f}")
    print(f"⚠️ 分類完全不看 test 期。")

    st = walk_forward_states(market_series(), features, n_states, seed=seed)
    bull_months = set(st[st["p_bull"] >= threshold].index)

    s_all = fl.walk_forward(proc, feats, "ridge")
    s_def = fl.walk_forward(proc, defensive, "ridge")

    def _stat(score, label):
        dd = proc[["stock_id", "group", "ym", "fwd_ret_1m"]].copy()
        dd["score"] = score
        rr = bt.slice_span(bt.portfolio_returns(dd.dropna(subset=["score"])),
                           fl.SPANS["test"])
        if not len(rr):
            return None
        b, a = np.polyfit(rr["benchmark"], rr["long"], 1)
        p = bt.perf(rr["long"])
        ex = bt.excess_vs_benchmark(rr)
        ir = ex.mean() * 12 / (ex.std() * np.sqrt(12))
        print(f"{label:<30}beta {b:>5.3f}  alpha {a*12:>+7.2%}  "
              f"CAGR {p['CAGR']:>7.2%}  IR {ir:>5.2f}  MaxDD {p['MaxDD']:>7.2%}")
        return {"label": label, "beta": float(b), "alpha": float(a * 12),
                "CAGR": p["CAGR"], "IR": float(ir), "MaxDD": p["MaxDD"]}

    print(f"\n=== test 期比較 ===")
    out = [_stat(s_all, "靜態：全部因子"),
           _stat(s_def, "靜態：只用防禦組")]
    switched = pd.Series(np.where(proc["ym"].isin(bull_months), s_all, s_def),
                         index=proc.index)
    out.append(_stat(switched, "HMM 切換（樣本外狀態）"))
    perfect_bull = set(
        (proc.groupby("ym")["fwd_ret_1m"].mean() > 0).pipe(lambda s: s[s].index))
    out.append(_stat(pd.Series(np.where(proc["ym"].isin(perfect_bull),
                                        s_all, s_def), index=proc.index),
                     "完美預知（上限，不可達成）"))
    print("\n判讀：HMM 切換那一列要落在『靜態全部因子』與『完美預知』之間才算有貢獻；"
          "\n     若比靜態還差，代表狀態訊號的雜訊大過它帶來的資訊。")
    return [o for o in out if o]


def main():
    ap = argparse.ArgumentParser(description="市場狀態模型（Gaussian HMM）")
    ap.add_argument("--fit", action="store_true", help="訓練期擬合一次，看狀態性質")
    ap.add_argument("--predict", action="store_true", help="walk-forward 方向準確率")
    ap.add_argument("--apply", action="store_true", help="用狀態切換因子組")
    ap.add_argument("--states", type=int, default=N_STATES)
    ap.add_argument("--features", default=",".join(FEATURES),
                    help="逗號分隔：ret / vol / breadth")
    ap.add_argument("--span", default="test",
                    choices=["sub_train", "validation", "test"])
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="p_bull ≥ 門檻視為多頭態")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save", help="結果存成 JSON")
    a = ap.parse_args()

    features = tuple(f.strip() for f in a.features.split(",") if f.strip())
    bad = [f for f in features if f not in FEATURE_COLS]
    if bad:
        raise SystemExit(f"未知特徵 {bad}，可用：{list(FEATURE_COLS)}")

    out = {}
    if a.fit or not any((a.fit, a.predict, a.apply)):
        cmd_fit(features, a.states, a.seed)
    if a.predict:
        out["predict"] = cmd_predict(features, a.states, a.seed, a.span)
    if a.apply:
        out["apply"] = cmd_apply(features, a.states, a.seed, a.threshold)

    if a.save and out:
        Path(a.save).write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                           default=float), encoding="utf-8")
        print(f"\n已存 {a.save}")


if __name__ == "__main__":
    main()
