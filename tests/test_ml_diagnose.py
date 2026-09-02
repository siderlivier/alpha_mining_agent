"""
過擬合／穩健性診斷套件（ml_diagnose）的正確性測試。

這套診斷的目的是「拆穿假的 IR」，所以它自己更不能算錯——一個有 bug 的
過擬合檢定會給出安心的假象，比沒有檢定還糟。每個統計量都用「已知答案的
合成資料」驗證：純噪音策略應該得到什麼、真有 edge 的策略應該得到什麼。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import backtest as bt          # noqa: E402
import ml_diagnose as md       # noqa: E402

pytest.importorskip("scipy")

STOCKS = [f"S{i:03d}" for i in range(40)]


def _months(n):
    return [str(p) for p in pd.period_range("2015-01", periods=n, freq="M")]


# ---------------------------------------------------------------------------
# 規模代理偵測：兩道判準
# ---------------------------------------------------------------------------

def test_formula_detects_size_fields():
    """公式裡直接出現 amt_21／turn_21 = 定義上的規模代理，不必等相關度達標。"""
    lib = {
        "F-001": {"formula": "cs_rank(sub(cs_rank(rev_yoy), cs_rank(amt_21)))"},
        "F-002": {"formula": "cs_rank(sdiv(px_hi252, turn_21))"},
        "F-003": {"formula": "cs_rank(ts_mean(roe, 12))"},
        "F-004": {"formula": None},                 # 參考因子沒有公式
        "R-001": {},                                # 欄位缺失也不能炸
    }
    hits = md.formula_size_proxies(lib, list(lib))
    assert set(hits) == {"F-001", "F-002"}


def test_correlation_detects_behavioural_size_proxy():
    """沒寫 amt_21、但行為上就是規模排序的因子，要靠相關度那道抓出來。"""
    rng = np.random.default_rng(3)
    ym = _months(24)
    rows = []
    for m in ym:
        amt = rng.lognormal(15, 1.5, len(STOCKS))
        rows.append(pd.DataFrame({
            "stock_id": STOCKS, "ym": m, "group": "電子",
            "amt_21": amt,
            "size_clone": np.log(amt) + rng.normal(0, 0.05, len(STOCKS)),
            "independent": rng.normal(size=len(STOCKS)),
        }))
    d = pd.concat(rows, ignore_index=True)

    mb = d[["stock_id", "ym", "amt_21"]]
    proc = d[["stock_id", "ym", "group", "size_clone", "independent"]]

    def fake_read(path, columns=None):
        return mb.copy()

    import pandas as _pd
    orig = _pd.read_parquet
    _pd.read_parquet = fake_read
    try:
        corr, hits = md.detect_size_proxies(proc, ["size_clone", "independent"])
    finally:
        _pd.read_parquet = orig

    assert hits == ["size_clone"], f"偵測結果不對：{hits}（corr={dict(corr)}）"
    assert corr["size_clone"] > 0.9
    assert abs(corr["independent"]) < 0.3


# ---------------------------------------------------------------------------
# 超額統計
# ---------------------------------------------------------------------------

def test_excess_stats_matches_hand_computation():
    n = 36
    ex = np.array([0.01] * (n // 2) + [-0.002] * (n // 2))
    rets = pd.DataFrame({"long": 0.02 + ex, "benchmark": [0.02] * n},
                        index=_months(n))
    s = md.excess_stats(rets)
    assert s["月數"] == n
    assert s["年化超額"] == pytest.approx(ex.mean() * 12)
    assert s["勝率"] == pytest.approx(0.5)
    sd = pd.Series(ex).std()
    assert s["IR"] == pytest.approx(ex.mean() * 12 / (sd * np.sqrt(12)))


def test_excess_stats_needs_minimum_months():
    rets = pd.DataFrame({"long": [0.01] * 3, "benchmark": [0.0] * 3},
                        index=_months(3))
    assert md.excess_stats(rets) == {}


# ---------------------------------------------------------------------------
# Deflated Sharpe
# ---------------------------------------------------------------------------

def _noise_rets(seed, n=120, mu=0.0, sd=0.03):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=_months(n))


def test_deflated_sharpe_punishes_more_trials():
    """
    同一條報酬序列，試驗數越多 → 去膨脹門檻越高 → DSR 越低。

    這正是 DSR 存在的意義：「我試了 5 個策略挑到這個」和「我試了 500 個
    策略挑到這個」，同樣的 Sharpe 可信度完全不同。
    """
    target = _noise_rets(1, mu=0.012)          # 有一點真 edge
    few = [_noise_rets(i).mean() / _noise_rets(i).std() for i in range(2, 8)]
    many = [_noise_rets(i).mean() / _noise_rets(i).std() for i in range(2, 80)]

    dsr_few, sr_a, sr0_few, n_few, _ = md.deflated_sharpe(target, few)
    dsr_many, sr_b, sr0_many, n_many, _ = md.deflated_sharpe(target, many)

    assert sr_a == sr_b, "目標策略的 Sharpe 不該隨試驗集改變"
    assert n_many > n_few
    assert sr0_many > sr0_few, "試驗數變多，期望最大 Sharpe 門檻必須提高"
    assert dsr_many < dsr_few, "試驗數變多，DSR 必須下降"


def test_deflated_sharpe_is_high_for_strong_edge_and_low_for_noise():
    trials = [_noise_rets(i).mean() / _noise_rets(i).std() for i in range(2, 40)]
    strong = md.deflated_sharpe(_noise_rets(101, mu=0.02), trials)[0]
    noise = md.deflated_sharpe(_noise_rets(102, mu=0.0), trials)[0]
    assert strong > 0.9, f"明顯有 edge 的策略 DSR 只有 {strong:.3f}"
    assert noise < 0.6, f"純噪音策略 DSR 高達 {noise:.3f}"
    assert 0.0 <= strong <= 1.0 and 0.0 <= noise <= 1.0


def test_deflated_sharpe_needs_enough_trials():
    with pytest.raises(SystemExit):
        md.deflated_sharpe(_noise_rets(1), [0.1])


# ---------------------------------------------------------------------------
# PBO / CSCV
# ---------------------------------------------------------------------------

def test_pbo_near_half_for_pure_noise():
    """
    全部都是純噪音的策略母體：樣本內最好的那個，到樣本外落在中位數以下的
    機率應該接近一半——這是 PBO 的零假設基準。抓不到 0.5 附近就代表實作有錯。
    """
    rng = np.random.default_rng(7)
    R = pd.DataFrame(rng.normal(0, 0.03, size=(96, 20)), index=_months(96))
    pbo, logits = md.cscv_pbo(R, S=8)
    assert 0.3 < pbo < 0.7, f"純噪音母體的 PBO 應該接近 0.5，實際 {pbo:.3f}"
    assert len(logits) == 70          # C(8,4)


def test_pbo_low_when_one_strategy_genuinely_dominates():
    """有一個策略是真的好（每一段都好）→ PBO 應該很低。"""
    rng = np.random.default_rng(11)
    R = pd.DataFrame(rng.normal(0, 0.03, size=(96, 20)), index=_months(96))
    R.iloc[:, 0] = rng.normal(0.03, 0.02, 96)      # 穩定領先
    pbo, _ = md.cscv_pbo(R, S=8)
    assert pbo < 0.1, f"真正穩健的策略不該被判為過擬合，PBO={pbo:.3f}"


def test_pbo_rejects_insufficient_sample():
    R = pd.DataFrame(np.random.normal(size=(10, 5)), index=_months(10))
    with pytest.raises(SystemExit):
        md.cscv_pbo(R, S=12)


# ---------------------------------------------------------------------------
# IC 定向
# ---------------------------------------------------------------------------

def test_mean_ic_sign_follows_relationship():
    """單因子策略要靠 IC 正負決定方向，否則所有負向因子都會被誤判為無效。"""
    rng = np.random.default_rng(5)
    rows = []
    for m in _months(24):
        s = rng.normal(size=len(STOCKS))
        rows.append(pd.DataFrame({
            "stock_id": STOCKS, "ym": m, "group": "電子",
            "score": s, "fwd_ret_1m": 0.02 * s + rng.normal(0, 0.01, len(STOCKS))}))
    pos = pd.concat(rows, ignore_index=True)
    assert md._mean_ic(pos) > 0.5
    neg = pos.assign(score=-pos["score"])
    assert md._mean_ic(neg) < -0.5


def test_mean_ic_skips_thin_industries():
    """產業內樣本不足 MIN_STOCKS 時要跳過，不能拿 3 檔股票算相關。"""
    rng = np.random.default_rng(2)
    few = STOCKS[:3]
    d = pd.DataFrame({"stock_id": few * 12,
                      "ym": sum([[m] * 3 for m in _months(12)], []),
                      "group": "電子",
                      "score": rng.normal(size=36),
                      "fwd_ret_1m": rng.normal(size=36)})
    assert np.isnan(md._mean_ic(d))


# ---------------------------------------------------------------------------
# 前瞻／切分紀律
# ---------------------------------------------------------------------------

def test_holdings_reproduce_backtest_selection():
    """
    `_holdings()` 重建的持股必須與 backtest 的選股口徑完全一致，
    否則產業組成與貢獻歸因講的是另一個投資組合。
    """
    rng = np.random.default_rng(4)
    rows = []
    for m in _months(6):
        for g, n in (("電子", 30), ("生技", 20), ("金融", 5)):   # 金融不足 8 檔
            for k in range(n):
                rows.append({"stock_id": f"{g}{k:02d}", "group": g, "ym": m,
                             "score": rng.normal(),
                             "fwd_ret_1m": rng.normal(0, 0.05)})
    d = pd.DataFrame(rows)
    H = md._holdings(d)

    assert "金融" not in set(H["group"]), "檔數不足的產業必須被跳過"
    per_month = H.groupby("ym").size()
    # 電子 30 檔 → int(30*0.10)=3；生技 20 檔 → 2
    assert (per_month == 5).all(), f"每月持股數不對：{per_month.to_dict()}"
    # 權重加總為 1，貢獻加總 = 組合報酬（未扣成本）
    for ym, g in H.groupby("ym"):
        assert g["w"].sum() == pytest.approx(1.0)
        assert g["contrib"].sum() == pytest.approx((g["fwd_ret_1m"] * g["w"]).sum())
    # 選到的必須是各產業內分數最高的
    for (ym, grp), g in H.groupby(["ym", "group"]):
        pool = d[(d["ym"] == ym) & (d["group"] == grp)]
        n = max(1, int(len(pool) * bt.TOP_Q))
        assert set(g["stock_id"]) == set(
            pool.nlargest(n, "score")["stock_id"])


def test_backtest_long_leg_is_long_only():
    """
    釐清一件容易誤會的事：`long` 這條腿**只做多**，沒有任何空頭部位。

    `portfolio_returns` 另外算了 `long_short`，但 factor_lab 與 ml_diagnose
    報的每一個數字都取 `long`。這條測試把這件事釘住——哪天有人改成用
    long_short 當 headline，測試會提醒他那是另一種策略。
    """
    rng = np.random.default_rng(6)
    rows = []
    for m in _months(12):
        for k in range(30):
            rows.append({"stock_id": f"E{k:02d}", "group": "電子", "ym": m,
                         "score": rng.normal(),
                         "fwd_ret_1m": rng.normal(0.01, 0.05)})
    d = pd.DataFrame(rows)
    r = bt.portfolio_returns(d, cost=0.0, min_stocks=8)

    # long 完全由多頭持股的報酬構成：手算一個月來對
    m0 = r.index[0]
    g = d[d["ym"] == m0]
    n = max(1, int(len(g) * bt.TOP_Q))
    expect = g.nlargest(n, "score")["fwd_ret_1m"].mean()
    assert r.loc[m0, "long"] == pytest.approx(expect)

    # long_short 一定不等於 long（除非空頭腿剛好為 0），且是另一條腿
    assert not np.allclose(r["long"], r["long_short"])
    short_leg = r["long_short"] - r["long"]
    assert short_leg.abs().sum() > 0, "long_short 應該含有空頭腿"


def test_span_mask_respects_config_split(monkeypatch):
    """所有診斷都必須只看指定切分期——預設是唯一真正樣本外的 test 期。"""
    import factor_lab as fl
    monkeypatch.setitem(fl.SPANS, "test", ("2015-06", "2015-09"))
    d = pd.DataFrame({"ym": _months(12), "x": range(12)})
    out = md.span_mask(d, "test")
    assert list(out["ym"]) == ["2015-06", "2015-07", "2015-08", "2015-09"]


def test_liquidity_percentile_is_within_month():
    """
    流動性百分位必須是「當月橫斷面」排名。

    用全期排名會有兩個問題：一是把跨期的市場規模成長混進來（2012 年的大型股
    可能不如 2026 年的中型股活絡），二是當月的排名會被未來月份的分布影響——
    那就是前瞻偏差。
    """
    rng = np.random.default_rng(9)
    ym = _months(6)
    rows = []
    for i, m in enumerate(ym):
        # 讓成交金額逐月整體放大，全期排名會被這個趨勢主導
        rows.append(pd.DataFrame({"stock_id": STOCKS, "ym": m, "group": "電子",
                                  "fwd_ret_1m": rng.normal(0, 0.05, len(STOCKS)),
                                  "score": rng.normal(size=len(STOCKS)),
                                  "amt_21": rng.lognormal(15, 1.0, len(STOCKS))
                                  * (10 ** i)}))
    d = pd.concat(rows, ignore_index=True)

    import pandas as _pd
    orig = _pd.read_parquet
    _pd.read_parquet = lambda p, columns=None: d[["stock_id", "ym", "amt_21"]].copy()
    try:
        out = md.attach_liquidity(d.drop(columns=["amt_21"]))
    finally:
        _pd.read_parquet = orig

    # 每個月的百分位分布都必須覆蓋 0~1，而不是被跨期趨勢壓成同一段
    per_month = out.groupby("ym")["liq_pct"].agg(["min", "max"])
    assert (per_month["min"] < 0.1).all() and (per_month["max"] > 0.9).all(), \
        f"百分位不是當月排名：\n{per_month}"


# ---------------------------------------------------------------------------
# 狀態條件式因子分析
# ---------------------------------------------------------------------------

def test_regime_ic_separates_defensive_from_offensive():
    """
    造兩個因子：一個只在大盤下跌月有效、一個只在上漲月有效。
    `factor_ic_by_regime` 必須把它們分到相反的兩端。

    這是狀態模型真正該接的地方（見 --regime-factors 的 docstring），
    所以分類邏輯本身不能算錯——分反了會讓人在下跌月押上攻擊型因子。
    """
    rng = np.random.default_rng(8)
    ym = _months(60)
    rows = []
    for i, m in enumerate(ym):
        n = len(STOCKS)
        # 前半年漲、後半年跌交替，確保兩種狀態都有足夠月數
        drift = 0.04 if i % 2 == 0 else -0.04
        defensive = rng.normal(size=n)
        offensive = rng.normal(size=n)
        noise = rng.normal(0, 0.02, n)
        # 下跌月時 defensive 有解釋力；上漲月時 offensive 有解釋力
        signal = (defensive if drift < 0 else offensive) * 0.05
        rows.append(pd.DataFrame({
            "stock_id": STOCKS, "ym": m, "group": "電子",
            "defensive": defensive, "offensive": offensive,
            "fwd_ret_1m": drift + signal + noise}))
    proc = pd.concat(rows, ignore_index=True)

    R = md.factor_ic_by_regime(proc, ["defensive", "offensive"],
                               ym[0], ym[-1], {})
    got = R.set_index("因子")["差異"]          # 差異 = ICIR_跌 − ICIR_漲
    assert got["defensive"] > 0.5, f"防禦型沒被認出來：{got.to_dict()}"
    assert got["offensive"] < -0.5, f"攻擊型沒被認出來：{got.to_dict()}"
    # 每個因子在自己的狀態下 ICIR 應該明顯為正
    d = R.set_index("因子")
    assert d.loc["defensive", "ICIR_跌"] > d.loc["defensive", "ICIR_漲"]
    assert d.loc["offensive", "ICIR_漲"] > d.loc["offensive", "ICIR_跌"]


def test_regime_ic_skips_factors_with_too_few_months():
    """月數不足的因子要被跳過，不能拿 10 個月算 ICIR。"""
    rng = np.random.default_rng(1)
    rows = []
    for m in _months(10):
        rows.append(pd.DataFrame({
            "stock_id": STOCKS, "ym": m, "group": "電子",
            "f": rng.normal(size=len(STOCKS)),
            "fwd_ret_1m": rng.normal(0, 0.05, len(STOCKS))}))
    proc = pd.concat(rows, ignore_index=True)
    R = md.factor_ic_by_regime(proc, ["f"], "2015-01", "2015-10", {})
    assert len(R) == 0
