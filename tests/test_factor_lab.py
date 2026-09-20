"""
因子組合實驗室（factor_lab）與回測（backtest）的正確性 + 前瞻偏差檢測。

這兩支是新加的「下游」模組：agent 挖出來的因子在這裡被合成成分數、跑成
組合績效。上游（DSL / eval_candidates）已經有前瞻測試，但合成這一層有它
自己的三個洩漏面，各對應一組測試：

  A. walk_forward 的切窗       → 訓練窗不得含測試期，且要留 embargo
  B. prep 的標準化             → 產業內 z-score 只能用「當期橫斷面」
  C. backtest 的持股/換手/成本 → 只吃算好的 score，不做跨期運算

方法同 test_pipeline_no_lookahead：造合成資料，複製一份把 cut 之後汙染成
極端值，斷言 cut 之前的輸出逐項相同（浮點完全相等，不給容差——真的沒讀到
未來的話，那些數字是同一串運算，位元級相同）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import backtest as bt          # noqa: E402
import factor_lab as fl        # noqa: E402

N_MONTHS = 96                  # 2012-01 ~ 2019-12
STOCKS = [f"S{i:03d}" for i in range(40)]
GROUPS = {s: ["半導體", "電子", "生技", "金融"][i % 4] for i, s in enumerate(STOCKS)}
FEATS = ["F-001", "F-002", "F-003"]


def test_load_panel_excludes_short_only(tmp_path, monkeypatch):
    from memory import Memory
    mem = Memory(tmp_path / "memory")
    mem.ensure()
    mem._save_library({"F-001": {"name_zh": "long", "trading_use": "long_only"},
                       "F-002": {"name_zh": "short", "trading_use": "short_only"}})
    pd.DataFrame({"factor_id": ["F-001", "F-002"], "ym": ["2015-01"] * 2,
                  "stock_id": ["A"] * 2, "value": [1., 2.]}).to_parquet(mem.values_path)
    (tmp_path / "data").mkdir()
    pd.DataFrame({"ym": ["2015-01"], "stock_id": ["A"], "group": ["G"],
                  "fwd_ret_1m": [0.1]}).to_parquet(tmp_path / "data/monthly_base.parquet")
    monkeypatch.setattr(fl, "Memory", lambda: mem)
    monkeypatch.setattr(fl, "ROOT", tmp_path)
    _, feats, _ = fl.load_panel("own")
    assert feats == ["F-001"]
    with pytest.raises(SystemExit, match="僅做空"):
        fl.load_panel(factors=["F-002"])
CORRUPT_AT = 71                # 汙染這個索引「之後」的月份


def _months(n=N_MONTHS):
    return [str(p) for p in pd.period_range("2012-01", periods=n, freq="M")]


def _make_panel(seed=11):
    """合成 [stock_id, group, ym, fwd_ret_1m, F-00x]；標籤與因子有真實關聯。"""
    rng = np.random.default_rng(seed)
    ym = _months()
    df = pd.DataFrame(
        [(s, m) for s in STOCKS for m in ym], columns=["stock_id", "ym"])
    df["group"] = df["stock_id"].map(GROUPS)
    n = len(df)
    for k, f in enumerate(FEATS):
        v = rng.normal(size=n)
        v[rng.random(n) < 0.05] = np.nan        # 保留缺值路徑
        df[f] = v
    signal = df[FEATS].fillna(0.0).mul([0.6, 0.3, -0.2]).sum(axis=1)
    df["fwd_ret_1m"] = 0.01 * signal + rng.normal(0, 0.06, n)
    return df.reset_index(drop=True)


def _corrupt(df, cut_ym):
    """把 cut_ym 之後的所有因子欄位換成極端值（標籤保持不變）。"""
    bad = df.copy()
    mask = bad["ym"] > cut_ym
    for f in FEATS:
        bad.loc[mask, f] = 1e7
    return bad


@pytest.fixture(scope="module")
def panels():
    base = _make_panel()
    cut = _months()[CORRUPT_AT]
    return base, _corrupt(base, cut), cut


@pytest.fixture(autouse=True)
def _fixed_hyper(monkeypatch):
    """把切窗參數釘死，測試不受 config.yaml 調整影響。"""
    monkeypatch.setattr(fl, "MIN_TRAIN_MONTHS", 48)
    monkeypatch.setattr(fl, "RETRAIN_EVERY", 12)
    monkeypatch.setattr(fl, "EMBARGO", 1)


# ---------------------------------------------------------------------------
# B. prep：標準化只能用當期橫斷面
# ---------------------------------------------------------------------------

def test_prep_is_cross_sectional_only(panels):
    """汙染未來 → cut 之前每一列的 z-score 與 y 必須完全不變。"""
    base, bad, cut = panels
    a = fl.prep(base, FEATS)
    b = fl.prep(bad, FEATS)
    ma, mb = a["ym"] <= cut, b["ym"] <= cut
    cols = FEATS + ["y"]
    pd.testing.assert_frame_equal(
        a.loc[ma, cols].reset_index(drop=True),
        b.loc[mb, cols].reset_index(drop=True),
        check_exact=True)


def test_prep_zscore_is_within_industry_month(panels):
    """z-score 的分組必須是 (ym, group)：每個 (月, 產業) 格內均值 ≈ 0。"""
    base, _, _ = panels
    p = fl.prep(base, FEATS)
    # fillna(0.0) 會把缺值拉回 0，故只檢查原本就有值的列
    orig = base["F-001"].notna().values
    mu = p.loc[orig].groupby(["ym", "group"])["F-001"].mean()
    assert mu.abs().max() < 1e-9, f"產業內去均值沒做乾淨，最大殘餘 {mu.abs().max()}"


def test_prep_does_not_use_full_period_stats(panels):
    """
    反向確認：若改用全期 mean/std，早期的數字一定會被未來改變。
    這個測試保護的是「不要圖方便改成全期標準化」——真改了，上面那支才會紅。
    """
    base, bad, cut = panels

    def full_period_z(df):
        out = df.copy()
        for c in FEATS:
            out[c] = (out[c] - out[c].mean()) / out[c].std()
        return out

    a, b = full_period_z(base), full_period_z(bad)
    ma = a["ym"] <= cut
    assert not np.allclose(a.loc[ma, FEATS].fillna(0).values,
                           b.loc[ma, FEATS].fillna(0).values), \
        "合成資料的汙染強度不足，測不出全期標準化的洩漏——請調高 1e7"


# ---------------------------------------------------------------------------
# A. walk_forward：切窗 + embargo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["equal", "ridge"])
def test_walk_forward_no_lookahead(panels, kind):
    """
    核心斷言：汙染 cut 之後的資料，cut 之前的樣本外分數必須逐列完全相同。

    分段 i 用 ym <= months[i-1-EMBARGO] 訓練、預測 months[i:i+12]。測試期
    整段落在 cut 之前的分段，其訓練窗也必然在 cut 之前，兩份資料算出來的
    分數應該位元級相同。任何一處讀到未來（例如訓練窗誤用全期、或標準化跨期），
    這個等式就會破。
    """
    base, bad, cut = panels
    pa = fl.prep(base, FEATS)
    pb = fl.prep(bad, FEATS)
    sa = fl.walk_forward(pa, FEATS, kind)
    sb = fl.walk_forward(pb, FEATS, kind)

    mask = (pa["ym"] <= cut) & sa.notna()
    assert mask.sum() > 0, "cut 之前沒有任何樣本外分數，測試沒有實際效力"
    pd.testing.assert_series_equal(sa[mask], sb[mask], check_exact=True)


def test_walk_forward_warmup_has_no_score(panels):
    """前 MIN_TRAIN_MONTHS 個月沒有足夠訓練資料，分數必須是 NaN。"""
    base, _, _ = panels
    p = fl.prep(base, FEATS)
    s = fl.walk_forward(p, FEATS, "ridge")
    warm = _months()[:fl.MIN_TRAIN_MONTHS]
    assert s[p["ym"].isin(warm)].isna().all(), "暖身期不該有分數"
    assert s[~p["ym"].isin(warm)].notna().any(), "暖身期之後應該要有分數"


def test_segments_embargo_gap():
    """
    訓練窗尾與測試窗頭之間必須隔 EMBARGO 個月。

    月頻標籤看下一個月（fwd_ret_1m），訓練期最後一個月的 y 其實用到了下個月的
    報酬；不留空檔，那個月就會和測試期第一個月重疊。
    """
    months = _months()
    segs = fl.segments(months)
    assert segs, "沒有切出任何訓練分段"
    for cut, test_months in segs:
        gap = months.index(test_months[0]) - months.index(cut)
        assert gap == fl.EMBARGO + 1, (
            f"訓練尾 {cut} 與測試頭 {test_months[0]} 只隔 {gap} 個月，"
            f"應為 {fl.EMBARGO + 1}")


def test_segments_train_window_never_touches_test():
    """訓練窗（ym <= cut）與該段測試月份不得有任何交集，且不得含未來月份。"""
    months = _months()
    segs = fl.segments(months)
    covered = []
    for cut, test_months in segs:
        train_months = [m for m in months if m <= cut]
        assert not (set(train_months) & set(test_months)), \
            f"訓練窗與測試窗重疊：cut={cut}, test={test_months[:2]}…"
        assert max(train_months) < min(test_months)
        covered += test_months
    # 測試段不重疊、且完整涵蓋暖身期之後的所有月份
    assert len(covered) == len(set(covered)), "測試段之間互相重疊"
    assert covered == months[fl.MIN_TRAIN_MONTHS:], "測試段沒有連續涵蓋暖身期之後"


def test_segments_train_window_expands(panels):
    """擴張窗：後面的分段訓練資料只會更多，不會更少。"""
    base, _, _ = panels
    p = fl.prep(base, FEATS)
    months = sorted(p["ym"].unique())
    sizes = [int((p["ym"] <= cut).sum()) for cut, _ in fl.segments(months)]
    assert sizes == sorted(sizes), f"訓練窗沒有單調擴張：{sizes}"
    assert sizes[0] >= 1000, "第一段訓練樣本不足門檻，walk_forward 會整段跳過"


# ---------------------------------------------------------------------------
# C. backtest：持股、換手、成本
# ---------------------------------------------------------------------------

def _bt_frame(seed=5, n_months=12, per_group=10):
    rng = np.random.default_rng(seed)
    ym = _months(n_months)
    rows = []
    for m in ym:
        for gi, g in enumerate(["半導體", "電子"]):
            for k in range(per_group):
                rows.append({"stock_id": f"{g}{k:02d}", "group": g, "ym": m,
                             "score": rng.normal(),
                             "fwd_ret_1m": rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_portfolio_picks_within_industry():
    """top_q 是「各產業內」取，不是全池取——兩個產業都要有持股。"""
    df = _bt_frame()
    # 讓「半導體」的分數整體遠高於「電子」；若是全池排序，持股會全在半導體
    df.loc[df["group"] == "半導體", "score"] += 100
    holdings = []
    for ym, g in df.groupby("ym"):
        sub = g.copy()
        picks = []
        for grp, gg in sub.groupby("group"):
            n = max(1, int(len(gg) * 0.10))
            picks += list(gg.sort_values("score", ascending=False)
                          .head(n)["group"])
        holdings.append(set(picks))
    assert all(h == {"半導體", "電子"} for h in holdings)

    rets = bt.portfolio_returns(df, top_q=0.10, cost=0.0, min_stocks=8)
    assert len(rets) == 12
    assert (rets["nhold"] == 2).all(), "各產業各取 1 檔，應共持 2 檔"


def test_portfolio_skips_thin_industries():
    """產業內不足 min_stocks 就整個產業跳過，不能濫竽充數。"""
    df = _bt_frame(per_group=10)
    thin = df["group"] == "電子"
    keep = df[thin]["stock_id"].unique()[:3]         # 電子只留 3 檔
    df = df[~thin | df["stock_id"].isin(keep)]
    rets = bt.portfolio_returns(df, top_q=0.10, cost=0.0, min_stocks=8)
    assert (rets["nhold"] == 1).all(), "電子檔數不足應被跳過，只剩半導體 1 檔"


def test_cost_is_deducted_by_turnover():
    """成本按換手比例扣：cost=0 與 cost=c 的差額 == c × turnover。"""
    df = _bt_frame(seed=9)
    free = bt.portfolio_returns(df, top_q=0.10, cost=0.0, min_stocks=8)
    paid = bt.portfolio_returns(df, top_q=0.10, cost=0.01, min_stocks=8)
    delta = free["long"] - paid["long"]
    np.testing.assert_allclose(delta.values, 0.01 * free["turnover"].values,
                               rtol=0, atol=1e-12)
    assert free["turnover"].iloc[0] == 1.0, "第一個月從空倉建倉，換手應為 1"


def test_backtest_is_pure_per_month(panels):
    """
    回測不得做任何跨期運算：把某個月之後的 score 全部打亂，
    該月之前的 long/benchmark 必須完全不變（turnover 只在交界那月受影響）。
    """
    df = _bt_frame(seed=3, n_months=24)
    cut = _months(24)[11]
    bad = df.copy()
    rng = np.random.default_rng(0)
    m = bad["ym"] > cut
    bad.loc[m, "score"] = rng.normal(size=int(m.sum())) * 100

    a = bt.portfolio_returns(df, top_q=0.10, cost=0.004, min_stocks=8)
    b = bt.portfolio_returns(bad, top_q=0.10, cost=0.004, min_stocks=8)
    early = [x for x in a.index if x <= cut]
    pd.testing.assert_frame_equal(a.loc[early, ["long", "benchmark", "nhold"]],
                                  b.loc[early, ["long", "benchmark", "nhold"]],
                                  check_exact=True)


def test_perf_requires_six_months():
    """樣本 < 6 個月的年化數字沒有意義，必須回空 dict 而不是硬算。"""
    r = pd.Series([0.01] * 5)
    assert bt.perf(r) == {}
    assert bt.perf(pd.Series([0.01] * 6))["Months"] == 6


def test_initial_equity_in_drawdown_and_missing_returns_rejected():
    assert bt.perf(pd.Series([-.5,0,0,0,0,0]))["MaxDD"]==-.5
    assert bt.perf(pd.Series([.1]*6))["MaxDD"]==0
    with pytest.raises(ValueError,match="missing"):
        bt.perf(pd.Series([.1]*6+[np.nan]))


def test_missing_future_return_cannot_replace_selected_stock():
    d=pd.DataFrame({"stock_id":[str(i) for i in range(20)],"ym":"2020-01",
                    "group":"G","score":np.arange(20),"fwd_ret_1m":.01})
    d.loc[19,"fwd_ret_1m"]=np.nan
    with pytest.raises(bt.MissingReturnError) as caught:bt.portfolio_returns(d)
    assert set(caught.value.long_holdings)=={"18","19"}
    assert caught.value.missing==["19"]
    # Even an unheld missing constituent must not silently change benchmark.
    d.loc[19,"fwd_ret_1m"]=.01;d.loc[10,"fwd_ret_1m"]=np.nan
    with pytest.raises(bt.MissingReturnError) as caught:bt.portfolio_returns(d)
    assert set(caught.value.long_holdings)=={"18","19"}


def test_load_panel_freezes_coverage_on_training_and_masks_scope(tmp_path,monkeypatch):
    from memory import Memory
    mem=Memory(tmp_path/"memory");mem.ensure()
    mem._save_library({"F-001":{"industry_scope":"G"},"F-002":{}})
    (tmp_path/"data").mkdir()
    base=pd.DataFrame({"stock_id":["A","B"]*2,"ym":["2015-01"]*2+["2020-01"]*2,
                       "group":["G","H"]*2,"fwd_ret_1m":[.1,.2,np.nan,.3]})
    base.to_parquet(tmp_path/"data/monthly_base.parquet")
    vals=pd.DataFrame({"factor_id":["F-001"]*4+["F-002"]*2,
                       "stock_id":["A","B"]*3,"ym":["2015-01"]*2+["2020-01"]*4,
                       "value":[1,999,2,999,1,2]})
    vals.to_parquet(mem.values_path)
    monkeypatch.setattr(fl,"Memory",lambda:mem);monkeypatch.setattr(fl,"ROOT",tmp_path)
    d,feats,_=fl.load_panel("own")
    assert feats==["F-001"] and len(d)==4
    assert d.loc[d.group.eq("H"),"F-001"].isna().all()
    vals.loc[vals.ym.eq("2020-01"),"value"]=np.nan;vals.to_parquet(mem.values_path)
    assert fl.load_panel("own")[1]==feats


def test_excess_stats_isolates_beta():
    """
    超額 = 多頭 − 當月等權基準。基準漲多少不影響超額。

    這條是為了擋一個真實的誤判：台股 test 期等權基準 CAGR 就有 24%，
    多頭組合 30%+ 的絕對報酬裡大部分是 beta，不是因子的功勞。
    """
    n = 24
    rets = pd.DataFrame({
        "long": [0.02] * n,
        "benchmark": [0.01] * n,
        "turnover": [0.3] * n,
        "nhold": [10] * n,
    }, index=_months(n))
    st = bt.excess_stats(rets)
    assert st["Excess"] == pytest.approx(0.01 * 12)
    assert st["ExcessWin"] == 1.0
    assert st["BenchCAGR"] > 0

    # 把基準與多頭同步抬高 → 超額不變，但 CAGR 兩邊都變高
    boom = rets.assign(long=rets["long"] + 0.05, benchmark=rets["benchmark"] + 0.05)
    st2 = bt.excess_stats(boom)
    assert st2["Excess"] == pytest.approx(st["Excess"])
    assert bt.perf(boom["long"])["CAGR"] > bt.perf(rets["long"])["CAGR"]


def test_evaluate_includes_excess_columns():
    """evaluate() 回傳的績效必須帶超額欄位，否則 --compare 的表會缺格。"""
    df = _bt_frame(seed=4, n_months=24)
    pf, turn = bt.evaluate(df, span=None, min_stocks=8)
    assert set(pf) >= {"CAGR", "Sharpe", "Excess", "IR", "ExcessWin", "BenchCAGR"}
    assert np.isfinite(turn)


def test_slice_span_is_inclusive():
    rets = pd.DataFrame({"long": np.arange(12.0)}, index=_months(12))
    s = bt.slice_span(rets, ("2012-03", "2012-06"))
    assert list(s.index) == ["2012-03", "2012-04", "2012-05", "2012-06"]
    assert len(bt.slice_span(rets, None)) == 12


# ---------------------------------------------------------------------------
# 端到端：分數 → 回測，且只在 test 期取數
# ---------------------------------------------------------------------------

def test_score_and_backtest_end_to_end(panels, monkeypatch):
    """整條 prep → walk_forward → 回測跑得通，且 span 真的有切到。"""
    base, _, _ = panels
    p = fl.prep(base, FEATS)
    monkeypatch.setitem(fl.SPANS, "test", ("2016-01", "2019-12"))
    pf = fl.score_and_backtest(p, FEATS, "ridge", span="test")
    assert pf, "端到端沒有產出績效"
    assert set(pf) >= {"CAGR", "Sharpe", "MaxDD", "WinRate", "Months",
                       "Turnover", "NFactors"}
    assert pf["NFactors"] == len(FEATS)
    assert pf["Months"] <= 48, "span 沒有生效，取到了 test 期以外的月份"


def test_greedy_selects_on_select_span_not_report_span(panels, monkeypatch, capsys):
    """
    貪婪選擇必須在 --select-span 上挑，在 --span 上回報。

    這是整支工具最容易自欺的地方：walk-forward 保證模型參數沒看到未來，但
    「要選哪幾個因子」本身也是一次擬合。用 test 期挑、又用 test 期報成績，
    等於拿答案卷挑答案——k 越大越好看，那是選擇偏誤不是 edge。
    """
    base, _, _ = panels
    monkeypatch.setattr(fl, "load_panel",
                        lambda which, factors=None: (base, list(FEATS),
                                                     {f: f for f in FEATS}))
    monkeypatch.setitem(fl.SPANS, "validation", ("2016-01", "2017-12"))
    monkeypatch.setitem(fl.SPANS, "test", ("2018-01", "2019-12"))

    seen = []
    real = fl.score_and_backtest

    def spy(proc, feats, kind, span="test"):
        seen.append(span)
        return real(proc, feats, kind, span)

    monkeypatch.setattr(fl, "score_and_backtest", spy)
    hist = fl.cmd_greedy("own", "equal", "test", 2, None,
                         rank="IR", select_span="validation")

    assert "validation" in seen, "沒有在挑選期評估過任何候選"
    assert "test" in seen, "沒有在回報期算出成績"
    # 候選搜尋（每一步試很多個因子）必須全在挑選期；回報期只在定案後算一次
    assert seen.count("validation") > seen.count("test")
    for h in hist:
        assert h["IR_挑選"] is not None and h["IR"] is not None
        assert h["IR_挑選"] != h["IR"], "兩期的數字一樣，select_span 可能沒生效"
    assert "回報期的數字才是誠實的" in capsys.readouterr().out


def test_greedy_warns_when_selecting_and_reporting_on_same_span(
        panels, monkeypatch, capsys):
    """沒指定 --select-span 時必須明確警告這是選擇偏誤。"""
    base, _, _ = panels
    monkeypatch.setattr(fl, "load_panel",
                        lambda which, factors=None: (base, list(FEATS),
                                                     {f: f for f in FEATS}))
    monkeypatch.setitem(fl.SPANS, "test", ("2016-01", "2019-12"))
    fl.cmd_greedy("own", "equal", "test", 2, None, rank="IR")
    out = capsys.readouterr().out
    assert "選擇偏誤" in out and "--select-span" in out


def test_cost_scan_decays_linearly_with_turnover(panels, monkeypatch, capsys):
    """
    成本掃描：超額必須隨成本單調下降，且衰退幅度 = 換手 × Δ成本 × 年化因子。

    這條等式成立的前提是「成本不進入最佳化」——持股由分數決定，成本只是事後
    扣款。等式若被打破，代表某處讓成本回頭影響了選股，那就不是純粹的敏感度
    測試了。
    """
    base, _, _ = panels
    monkeypatch.setattr(fl, "load_panel",
                        lambda which, factors=None: (base, list(FEATS), {}))
    monkeypatch.setitem(fl.SPANS, "test", ("2016-01", "2019-12"))
    costs = (0.002, 0.004, 0.008)
    rows = fl.cmd_cost_scan("own", ["ridge"], "test", None, costs=costs)

    r = rows[0]
    vals = [r[f"cost_{c}"] for c in costs]
    assert vals == sorted(vals, reverse=True), f"超額沒有隨成本單調下降：{vals}"
    expected = r["換手"] * (costs[-1] - costs[0]) * bt.ANN
    assert r["衰退"] == pytest.approx(expected, rel=1e-6), (
        f"衰退 {r['衰退']:.4%} ≠ 換手×Δ成本×12 = {expected:.4%}"
        "——成本可能不小心影響了選股")
    assert "換手" in capsys.readouterr().out


def test_score_and_backtest_returns_empty_when_span_too_short(panels, monkeypatch):
    """span 內不足 6 個月時回空 dict，不能讓 _print_table 撞 KeyError。"""
    base, _, _ = panels
    p = fl.prep(base, FEATS)
    monkeypatch.setitem(fl.SPANS, "test", ("2019-09", "2019-12"))
    assert fl.score_and_backtest(p, FEATS, "equal", span="test") == {}
