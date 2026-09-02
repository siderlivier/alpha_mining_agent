"""
市場狀態模型（hmm_regime）的前瞻偏差與正確性測試。

HMM 用在回測上有三個特別容易洩漏未來的地方，這支逐一釘住：

  1. **觀測值的時點對齊**——`fwd_ret_1m` 是未來報酬，拿它當「當期特徵」
     就是直接看答案。
  2. **推論用平滑而非濾波**——`hmmlearn.predict()`（Viterbi）與
     `predict_proba()`（平滑）都會用整段序列回推每個時點的狀態。
  3. **狀態標籤用測試期決定**——「哪個狀態是多頭」若靠測試期表現決定，
     等於拿答案卷貼標籤。

方法與 test_factor_lab 一致：汙染 cut 之後的資料，斷言 cut 之前的輸出
位元級不變（`check_exact=True`，不給容差）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("hmmlearn")
import hmm_regime as hr        # noqa: E402


def _months(n, start="2012-01"):
    return [str(p) for p in pd.period_range(start, periods=n, freq="M")]


def _two_regime_obs(n=180, seed=0):
    """造一段有兩種明顯狀態（低波動/高波動）的觀測序列。"""
    rng = np.random.default_rng(seed)
    x, state = [], 0
    for _ in range(n):
        state = state if rng.random() < 0.9 else 1 - state
        x.append(rng.normal(0.012 if state == 0 else -0.005,
                            0.02 if state == 0 else 0.07))
    return np.array(x).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 1. 觀測值的時點對齊
# ---------------------------------------------------------------------------

def test_market_series_lags_observations_by_one_month(monkeypatch, tmp_path):
    """
    `obs_ret[t]` 必須等於 `mkt[t-1]`。

    `mkt[t]` 是「t → t+1」的報酬，在月底 t 做決策時**還沒發生**。
    少 shift 一格，模型就直接看到了要預測的東西，準確率會假性飆高。
    """
    ym = _months(8)
    rows = []
    for i, m in enumerate(ym):
        for k in range(20):
            rows.append({"stock_id": f"S{k}", "ym": m,
                         "fwd_ret_1m": 0.01 * (i + 1)})   # 每月一個明確的值
    mb = pd.DataFrame(rows)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: mb.copy())

    d = hr.market_series()
    np.testing.assert_allclose(d["mkt"].values, [0.01 * (i + 1) for i in range(8)])
    assert np.isnan(d["obs_ret"].iloc[0]), "第一個月沒有前一期，必須是 NaN"
    np.testing.assert_allclose(d["obs_ret"].values[1:], d["mkt"].values[:-1],
                               rtol=0, atol=0)
    # breadth 同樣要落後一期
    assert np.isnan(d["obs_breadth"].iloc[0])


def test_observation_matrix_drops_warmup_rows(monkeypatch):
    ym = _months(12)
    mb = pd.DataFrame([{"stock_id": "S1", "ym": m, "fwd_ret_1m": 0.01}
                       for m in ym])
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: mb.copy())
    d = hr.market_series()
    X, idx = hr.observation_matrix(d, ("ret", "vol"))
    assert len(X) == len(idx)
    assert not np.isnan(X).any(), "暖身期的缺值列必須被丟掉"
    assert idx[0] > ym[0], "第一列不可能是第一個月（obs 落後一期 + 波動窗）"


# ---------------------------------------------------------------------------
# 2. 前向濾波：不得使用未來觀測
# ---------------------------------------------------------------------------

def test_forward_filter_is_causal():
    """
    前 k 步的濾波機率，不能因為「後面又多了觀測」而改變。

    這是濾波與平滑的分界線。若哪天有人把 forward_filter 換成
    `model.predict_proba()`（平滑），這條測試會立刻紅。
    """
    X = _two_regime_obs()
    model = hr.fit_hmm(X[:120], n_states=2, seed=0)
    short = hr.forward_filter(model, X[:80])
    long = hr.forward_filter(model, X)
    np.testing.assert_array_equal(short, long[:80])


def test_forward_filter_differs_from_smoothing():
    """
    反向確認：平滑機率**會**被未來觀測改變。

    這條測試保護的是「不要圖方便改用 predict_proba」——真改了，
    上面那條 causal 測試才會紅。若兩者其實相同，上面那條就是空的。
    """
    X = _two_regime_obs()
    model = hr.fit_hmm(X[:120], n_states=2, seed=0)
    smooth_short = model.predict_proba(X[:80])
    smooth_long = model.predict_proba(X)
    assert not np.allclose(smooth_short, smooth_long[:80]), \
        "合成資料的狀態太確定，測不出平滑與濾波的差別——請調高狀態重疊度"


def test_forward_filter_returns_valid_distribution():
    X = _two_regime_obs()
    model = hr.fit_hmm(X[:120], n_states=2, seed=0)
    alpha = hr.forward_filter(model, X)
    assert alpha.shape == (len(X), 2)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, rtol=0, atol=1e-9)
    assert (alpha >= 0).all()


# ---------------------------------------------------------------------------
# 3. 狀態標籤只能用訓練資料決定
# ---------------------------------------------------------------------------

def test_bull_state_uses_only_realised_pairs():
    """
    判定多頭態時，最後一個月的「後續報酬」還沒實現，不可以拿來用。

    觀測 X[j] 對應月份 idx[j]，而 mkt[idx[j]] 要到下個月底才知道。
    擬合點在 idx[n-1] 時，可用的配對只到 j = n-2。把最後一格的 mkt
    換成極端值，判定結果必須不變。
    """
    X = _two_regime_obs()
    idx = pd.Index(_months(len(X)))
    rng = np.random.default_rng(1)
    mkt = pd.Series(rng.normal(0.01, 0.05, len(X)), index=idx)
    model = hr.fit_hmm(X[:120], n_states=2, seed=0)

    a = hr.bull_state_from_train(model, X, idx, mkt, 120)
    poisoned = mkt.copy()
    poisoned.iloc[119] = 1e6            # 擬合點當期的 mkt：尚未實現
    b = hr.bull_state_from_train(model, X, idx, poisoned, 120)
    assert a == b, "判定用到了尚未實現的當期報酬"

    # 而更早的月份是已實現的，改動它**應該**會影響判定
    poisoned2 = mkt.copy()
    poisoned2.iloc[:118] = np.where(
        hr.forward_filter(model, X[:118]).argmax(axis=1) == a, -1.0, 1.0)
    c = hr.bull_state_from_train(model, X, idx, poisoned2, 120)
    assert c != a, "已實現月份的報酬被完全反轉，判定卻沒變——邏輯可能沒生效"


def test_label_states_is_stable_ordering():
    """label_states 只負責排序命名，必須與觀測均值一致。"""
    X = _two_regime_obs()
    model = hr.fit_hmm(X, n_states=2, seed=0)
    rank = hr.label_states(model)
    lo = min(rank, key=lambda k: rank[k])
    hi = max(rank, key=lambda k: rank[k])
    assert model.means_[lo, 0] <= model.means_[hi, 0]


# ---------------------------------------------------------------------------
# 4. walk-forward：整條路徑不得看未來
# ---------------------------------------------------------------------------

def _series_from(x, ym):
    """把一段市場報酬包成 market_series() 的輸出格式。"""
    d = pd.DataFrame({"mkt": x}, index=ym)
    d["obs_ret"] = d["mkt"].shift(1)
    d["obs_vol"] = d["obs_ret"].rolling(hr.VOL_WINDOW, min_periods=3).std()
    d["obs_breadth"] = (d["obs_ret"] > 0).astype(float)
    return d


def test_walk_forward_states_no_lookahead():
    """
    汙染 cut 之後的市場報酬，cut 之前的樣本外狀態機率必須逐格完全相同。

    這條同時檢查了三件事：參數擬合窗、前向濾波、多頭態判定，
    任何一處讀到未來，等式都會破。
    """
    n = 200
    x = _two_regime_obs(n).ravel()
    ym = _months(n)
    clean = _series_from(x, ym)

    cut_i = 150
    bad_x = x.copy()
    bad_x[cut_i:] = 5.0                     # 未來全汙染成極端值
    dirty = _series_from(bad_x, ym)

    kw = dict(features=("ret", "vol"), n_states=2, min_train=60,
              refit_every=12, seed=0)
    a = hr.walk_forward_states(clean, **kw)
    b = hr.walk_forward_states(dirty, **kw)

    cut_ym = ym[cut_i]
    ea = a[a.index < cut_ym]["p_bull"]
    eb = b[b.index < cut_ym]["p_bull"]
    assert len(ea) > 20, "cut 之前的樣本外月份太少，測試沒有實際效力"
    pd.testing.assert_series_equal(ea, eb, check_exact=True)


def test_walk_forward_fit_window_never_includes_prediction_month():
    """每一格的預測月份，都必須晚於該段擬合窗的最後一個月。"""
    n = 160
    d = _series_from(_two_regime_obs(n).ravel(), _months(n))
    st = hr.walk_forward_states(d, features=("ret", "vol"), n_states=2,
                                min_train=60, refit_every=12, seed=0)
    assert len(st) > 0
    for ym, row in st.iterrows():
        assert row["fit_upto"] < ym, (
            f"預測 {ym} 卻用了到 {row['fit_upto']} 的資料擬合")


def test_walk_forward_warmup_has_no_states():
    """前 min_train 個月沒有樣本外狀態。"""
    n = 160
    d = _series_from(_two_regime_obs(n).ravel(), _months(n))
    X, idx = hr.observation_matrix(d, ("ret", "vol"))
    st = hr.walk_forward_states(d, features=("ret", "vol"), n_states=2,
                                min_train=60, refit_every=12, seed=0)
    assert st.index[0] == idx[60]
    assert len(st) == len(X) - 60
