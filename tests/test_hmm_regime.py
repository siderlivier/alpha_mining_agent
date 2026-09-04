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


# ---------------------------------------------------------------------------
# 5. 總經特徵：只能透過 as_of() 取值
# ---------------------------------------------------------------------------

def test_attach_macro_uses_as_of_not_raw_join(monkeypatch, tmp_path):
    """
    總經特徵**必須**走 fetch_macro.as_of()（只回 pub_ym <= 決策月的資料），
    不能直接依 ym 併過來。

    景氣燈號要到次月底才公布，直接 join 就會讓模型在月底 t 看到當月的燈號
    ——那是還沒發生的資訊。這條測試把「差一個月」這件事釘死。
    """
    import fetch_macro as fm
    ym = _months(8, "2024-01")
    rows = [{"ym": m, "field": "monitoring_score", "value": 20.0 + i}
            for i, m in enumerate(ym)]
    panel = fm.build_panel([pd.DataFrame(rows)])
    p = tmp_path / "macro.parquet"
    panel.to_parquet(p, index=False)
    monkeypatch.setattr(hr, "MACRO_PATH", p)

    d = pd.DataFrame({"mkt": np.zeros(len(ym))}, index=ym)
    out = hr.attach_macro(d, ("monitoring_score",))

    # 2024-03 決策時只能看到 2024-02 的分數（21.0），不是 2024-03 的（22.0）
    assert out.loc["2024-03", "monitoring_score"] == 21.0
    raw = {r["ym"]: r["value"] for r in rows}
    assert out.loc["2024-03", "monitoring_score"] != raw["2024-03"], \
        "拿到了當月尚未公布的景氣分數——attach_macro 沒走 as_of()"
    # 整條序列都要落後一格
    got = out["monitoring_score"].dropna()
    for m, v in got.items():
        prev = str(pd.Period(m, freq="M") - 1)
        assert v == raw.get(prev), f"{m} 應該對到 {prev} 的值"


def test_attach_macro_noop_without_macro_features(tmp_path, monkeypatch):
    """沒指定總經特徵時不該去碰檔案（純價格模式不應依賴總經資料）。"""
    monkeypatch.setattr(hr, "MACRO_PATH", tmp_path / "does_not_exist.parquet")
    d = pd.DataFrame({"mkt": [0.0, 0.1]}, index=_months(2))
    out = hr.attach_macro(d, ("ret", "vol"))
    pd.testing.assert_frame_equal(out, d)


def test_attach_macro_errors_clearly_when_field_missing(monkeypatch, tmp_path):
    """指定的總經欄位不在面板裡時，要說清楚有哪些、該怎麼補。"""
    import fetch_macro as fm
    panel = fm.build_panel([pd.DataFrame(
        [{"ym": "2024-01", "field": "us_curve", "value": 0.01}])])
    p = tmp_path / "m.parquet"
    panel.to_parquet(p, index=False)
    monkeypatch.setattr(hr, "MACRO_PATH", p)
    with pytest.raises(SystemExit, match="monitoring_score"):
        hr.attach_macro(pd.DataFrame({"mkt": [0.0]}, index=["2024-01"]),
                        ("monitoring_score",))


def test_observation_matrix_accepts_macro_column_names():
    """總經欄位直接用 field 名當欄名，不經過 FEATURE_COLS 對照表。"""
    ym = _months(6)
    d = pd.DataFrame({"obs_ret": np.linspace(0, 0.05, 6),
                      "monitoring_score": np.arange(6.0)}, index=ym)
    X, idx = hr.observation_matrix(d, ("ret", "monitoring_score"))
    assert X.shape == (6, 2)
    assert list(idx) == ym


def test_observation_matrix_reports_unknown_column():
    d = pd.DataFrame({"obs_ret": [0.1, 0.2]}, index=_months(2))
    with pytest.raises(SystemExit, match="沒有這些欄位"):
        hr.observation_matrix(d, ("ret", "not_a_column"))


# ---------------------------------------------------------------------------
# --riskadj：風險調整後對照的三個數學性質
# ---------------------------------------------------------------------------

def _fake_returns(n=60, seed=0):
    """造一段有 beta 的月報酬：long = 0.8 × bench + alpha + 噪音。"""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    bench = pd.Series(rng.normal(0.01, 0.05, n),
                      index=[f"20{10 + i // 12:02d}-{i % 12 + 1:02d}"
                             for i in range(n)])
    long = 0.8 * bench + 0.004 + rng.normal(0, 0.02, n)
    return pd.DataFrame({"long": long, "benchmark": bench})


def test_sharpe_對槓桿免疫():
    """
    這是整個 --riskadj 論證的地基：Sharpe 加了槓桿不會變。

    若這條不成立，「用 Sharpe 當公正裁判」的說法就垮了——
    所以它值得一個專門的測試，而不是靠註解宣稱。
    """
    import hmm_regime as hr
    rr = _fake_returns()
    a = hr._full_metrics(rr, "x", lever=1.0)
    b = hr._full_metrics(rr, "x", lever=1.75)
    assert abs(a["Sharpe"] - b["Sharpe"]) < 1e-9, (a["Sharpe"], b["Sharpe"])


def test_槓桿讓_beta_與波動同比例放大():
    """beta 與 Vol 必須恰好乘上 k——這是「等 beta 對照」能成立的前提。"""
    import hmm_regime as hr
    rr = _fake_returns()
    k = 1.75
    a = hr._full_metrics(rr, "x", lever=1.0)
    b = hr._full_metrics(rr, "x", lever=k)
    assert abs(b["beta"] - a["beta"] * k) < 1e-9
    assert abs(b["Vol"] - a["Vol"] * k) < 1e-9


def test_IR_會因為降低曝險而扣分():
    """
    用錯裁判的機制本身也要被測到，否則 16.1 節的論證只是嘴上說說。

    把一個**完全沒有技術差異**的組合（同一條報酬序列）縮小曝險，
    Sharpe 不動，IR 卻掉——這就是 IR 對 beta 決策有系統性偏見的證明。
    """
    import hmm_regime as hr
    rr = _fake_returns()
    full = hr._full_metrics(rr, "x", lever=1.0)
    shrunk = hr._full_metrics(rr, "x", lever=0.6)
    assert abs(full["Sharpe"] - shrunk["Sharpe"]) < 1e-9
    assert shrunk["IR"] < full["IR"], (shrunk["IR"], full["IR"])


def test_超額可由_alpha_與_beta_貢獻分解():
    """超額 ≈ alpha + (beta−1)×基準。誤差要在 1pp 內（幾何 vs 算術的落差）。"""
    import hmm_regime as hr
    rr = _fake_returns(n=120)
    m = hr._full_metrics(rr, "x")
    assert abs(m["Excess"] - (m["alpha"] + m["BetaExcess"])) < 0.01


def test_自助法在同一條序列上給出零差異():
    """把兩條一模一樣的序列丟進去，ΔSharpe 必須恆為 0、CI 也是 0。"""
    import hmm_regime as hr
    rr = _fake_returns()
    b = hr._paired_bootstrap(rr["long"], rr["long"], rr["benchmark"],
                             n_boot=200, seed=0)
    assert abs(b["dSharpe"]) < 1e-9
    assert abs(b["dSharpe_lo"]) < 1e-9 and abs(b["dSharpe_hi"]) < 1e-9
    assert b["n"] == len(rr)
