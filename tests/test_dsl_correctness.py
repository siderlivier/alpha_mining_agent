"""
M0 驗收：DSL 計算結果 vs pandas 手算逐格一致（規格書 11 章 M0 驗收標準）。
"""
import numpy as np
import pandas as pd
import pytest

import dsl


def test_delay(data, group_map):
    got = dsl.compute("delay(f1, 3)", data, group_map)
    pd.testing.assert_frame_equal(got, data["f1"].shift(3))


def test_delta(data, group_map):
    got = dsl.compute("delta(f1, 12)", data, group_map)
    pd.testing.assert_frame_equal(got, data["f1"] - data["f1"].shift(12))


def test_ts_mean(data, group_map):
    got = dsl.compute("ts_mean(f1, 6)", data, group_map)
    exp = data["f1"].rolling(6, min_periods=4).mean()
    pd.testing.assert_frame_equal(got, exp)


def test_ts_std_max_min_med(data, group_map):
    for op, fn in [("ts_std", "std"), ("ts_max", "max"),
                   ("ts_min", "min"), ("ts_med", "median")]:
        got = dsl.compute(f"{op}(f2, 12)", data, group_map)
        exp = getattr(data["f2"].rolling(12, min_periods=8), fn)()
        pd.testing.assert_frame_equal(got, exp, obj=op)


def test_cs_rank_all(data, group_map):
    got = dsl.compute("cs_rank_all(f1)", data, group_map)
    pd.testing.assert_frame_equal(got, data["f1"].rank(axis=1, pct=True))


def test_cs_rank_groupwise(data, group_map):
    """產業內排名：手工按 group 分組 rank 對照。"""
    got = dsl.compute("cs_rank(f1)", data, group_map)
    exp = pd.DataFrame(np.nan, index=data["f1"].index, columns=data["f1"].columns)
    groups = {}
    for sid, g in group_map.items():
        groups.setdefault(g, []).append(sid)
    for g, cols in groups.items():
        exp[cols] = data["f1"][cols].rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(got, exp)


def test_ts_rank_manual():
    """小案例手算：序列 [1,3,2,5,4]，窗口 3。"""
    idx = pd.period_range("2020-01", periods=5, freq="M")
    df = pd.DataFrame({"A": [1.0, 3.0, 2.0, 5.0, 4.0]}, index=idx)
    got = dsl.compute("ts_rank(f1, 3)", {"f1": df}, {"A": "g"})
    # t2: [1,3,2] -> 2 是第 2 名/3；t3: [3,2,5] -> 5 是 3/3；t4: [2,5,4] -> 4 是 2/3
    exp = [np.nan, 2 / 2, 2 / 3, 3 / 3, 2 / 3]  # t1: [1,3] mp=2 -> 3 是 2/2
    np.testing.assert_allclose(got["A"].to_numpy(), exp, rtol=1e-12)


def test_sdiv_zero_denominator():
    idx = pd.period_range("2020-01", periods=3, freq="M")
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    y = pd.DataFrame({"A": [2.0, 0.0, -4.0]}, index=idx)
    got = dsl.compute("sdiv(f1, f2)", {"f1": x, "f2": y}, {"A": "g"})
    np.testing.assert_allclose(got["A"].to_numpy(), [0.5, np.nan, -0.75])


def test_sign_neg_composite(data, group_map):
    got = dsl.compute("neg(sign(f1))", data, group_map)
    exp = -np.sign(data["f1"])
    pd.testing.assert_frame_equal(got, exp)


def test_if_else_where(data, group_map):
    got = dsl.compute("if_else(f1 > 0, f2, neg(f2))", data, group_map)
    f1, f2 = data["f1"], data["f2"]
    # 語義：cond 為 NaN -> NaN；分支值的 NaN 自然傳遞
    exp = f2.where((f1 > 0), -f2).mask(f1.isna())
    pd.testing.assert_frame_equal(got, exp)


def test_ts_slope_perfect_line():
    """完美直線 y=2t+1 的斜率必為 2、R²=1、殘差=0。"""
    idx = pd.period_range("2020-01", periods=24, freq="M")
    df = pd.DataFrame({"A": 2.0 * np.arange(24) + 1.0}, index=idx)
    d = {"f1": df}
    gm = {"A": "g"}
    slope = dsl.compute("ts_slope(f1, 12)", d, gm)["A"].iloc[-1]
    rsq = dsl.compute("ts_rsq(f1, 12)", d, gm)["A"].iloc[-1]
    resi = dsl.compute("ts_resi(f1, 12)", d, gm)["A"].iloc[-1]
    np.testing.assert_allclose([slope, rsq, resi], [2.0, 1.0, 0.0], atol=1e-10)


def test_ts_corr(data, group_map):
    got = dsl.compute("ts_corr(f1, f2, 12)", data, group_map)
    exp = data["f1"].rolling(12, min_periods=8).corr(data["f2"])
    pd.testing.assert_frame_equal(got, exp)


def test_log1p_abs(data, group_map):
    got = dsl.compute("log1p_abs(f1)", data, group_map)
    x = data["f1"]
    exp = np.sign(x) * np.log1p(x.abs())
    pd.testing.assert_frame_equal(got, exp)


# ---------------------------------------------------------------------------
# 2026-08-19 人工核准的三個運算子提案
# ---------------------------------------------------------------------------

def test_rank_nz_excludes_zeros():
    """rank_nz：0 與 NaN 排除在排名之外，其餘在產業內重新排名。"""
    idx = pd.period_range("2020-01", periods=1, freq="M")
    cols = ["A", "B", "C", "D", "E"]
    df = pd.DataFrame([[1.0, 0.0, 3.0, np.nan, 2.0]], index=idx, columns=cols)
    gm = {c: "g" for c in cols}
    got = dsl.compute("rank_nz(f1)", {"f1": df}, gm).iloc[0]
    # 有效樣本只有 A=1, C=3, E=2 → 百分位 1/3, 3/3, 2/3
    np.testing.assert_allclose([got["A"], got["C"], got["E"]], [1 / 3, 1.0, 2 / 3])
    assert np.isnan(got["B"]) and np.isnan(got["D"]), "0 與 NaN 都要是 NaN"

    # 對照 cs_rank：0 會被納入排名，把有效樣本的百分位整個壓縮
    cs = dsl.compute("cs_rank(f1)", {"f1": df}, gm).iloc[0]
    assert cs["B"] == 0.25 and cs["C"] == 1.0
    assert cs["A"] != got["A"], "這正是 rank_nz 要解決的稀釋問題"


def test_industry_demean():
    """industry_demean：每期在各自產業內減去該產業均值，保留量級。"""
    idx = pd.period_range("2020-01", periods=2, freq="M")
    cols = ["A", "B", "C", "D"]
    df = pd.DataFrame([[1.0, 3.0, 10.0, 20.0],
                       [2.0, 2.0, 5.0, 15.0]], index=idx, columns=cols)
    gm = {"A": "g1", "B": "g1", "C": "g2", "D": "g2"}
    got = dsl.compute("industry_demean(f1)", {"f1": df}, gm)
    # g1 均值 2.0 / 2.0；g2 均值 15.0 / 10.0
    exp = pd.DataFrame([[-1.0, 1.0, -5.0, 5.0],
                        [0.0, 0.0, -5.0, 5.0]], index=idx, columns=cols)
    pd.testing.assert_frame_equal(got, exp)
    # 每期每產業內總和為 0
    np.testing.assert_allclose(got[["A", "B"]].sum(axis=1), [0.0, 0.0], atol=1e-12)


def test_streak_counts_consecutive_same_direction():
    """streak：從當期往回數，單期變化與當期同號的連續期數。"""
    #      idx:    0     1     2     3     4     5     6     7     8     9
    vals = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 15.0, 16.0, 16.0]
    # 變化:    -    +1    +1    +1    +1    +1    -1    +1    +1     0
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    df = pd.DataFrame({"A": vals}, index=idx)
    got = dsl.compute("streak(f1, 6)", {"f1": df}, {"A": "g"})["A"].tolist()

    # 暖機期：sign 序列首格為 NaN，_mp(6)=4，故前 4 格湊不滿最低樣本數
    assert all(np.isnan(v) for v in got[:4]), got[:4]
    assert got[4] == 4.0, "連續 4 期上升"
    assert got[5] == 5.0, "連續 5 期上升"
    assert got[6] == 1.0, "方向反轉，當期只算 1"
    assert got[7] == 1.0, "再反轉回來，重新從 1 起算"
    assert got[8] == 2.0, "連續 2 期上升"
    assert got[9] == 0.0, "當期無變化 → 0（方向未定義）"


def test_streak_true_counts_consecutive_true():
    """streak_true：條件連續為真的期數；一中斷即重置。"""
    #        idx:  0   1   2   3   4   5   6   7   8   9
    vals = [1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, 7.0, -2.0, 8.0]
    # >0 ?    T   T   T    F    T   T   T   T    F   T
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    df = pd.DataFrame({"A": vals}, index=idx)
    got = dsl.compute("streak_true(greater(f1, 0), 6)",
                      {"f1": df}, {"A": "g"})["A"].tolist()
    assert all(np.isnan(v) for v in got[:3]), f"暖機期（_mp(6)=4）{got[:3]}"
    assert got[3] == 0.0, "條件為假 → 0"
    assert got[4] == 1.0 and got[5] == 2.0 and got[7] == 4.0, "重新起算並累加"
    assert got[8] == 0.0, "再度中斷 → 0"
    assert got[9] == 1.0, "中斷後重新從 1 起算"


def test_streak_true_differs_from_streak():
    """streak_true(cond,n) 與 streak(x,n) 語意不同——這是分開實作的理由。"""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    d = {"f1": pd.DataFrame({"A": vals}, index=idx)}
    gm = {"A": "g"}
    a = dsl.compute("streak(f1, 6)", d, gm)["A"]          # 變化方向連續同號
    b = dsl.compute("streak_true(greater(f1, 0), 6)", d, gm)["A"]  # 條件連續為真
    # 單調上升且全為正：方向連續數會被 delta 的首格 NaN 拖慢一期
    assert not a.equals(b), "兩者不應相同"
    assert b.iloc[-1] == 6.0, "全期為正 → 上限 6"


def test_clip_std_clips_outliers():
    """clip_std：離群值被截到「前 n 期」的 mean±3σ，正常值與暖機期不動。"""
    rng = np.random.default_rng(11)
    vals = list(rng.normal(10.0, 0.3, 20)) + [1000.0]   # 最後一期是離群值
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    df = pd.DataFrame({"A": vals}, index=idx)
    got = dsl.compute("clip_std(f1, 12)", {"f1": df}, {"A": "g"})["A"]

    # 邊界用「前 n 期（不含當期）」——否則離群值會撐高自己的上界
    prev = df["A"].shift(1)
    upper = (prev.rolling(12, min_periods=8).mean()
             + 3 * prev.rolling(12, min_periods=8).std()).iloc[-1]
    assert got.iloc[-1] == pytest.approx(upper), "離群值應被截到上界"
    assert got.iloc[-1] < 20.0, f"沒有截尾：{got.iloc[-1]}"
    assert got.iloc[:8].tolist() == vals[:8], "暖機期不該改值"
    # 常態範圍內的值不該被動到
    normal = got.iloc[8:20]
    assert np.allclose(normal, vals[8:20]), "3σ 內的正常值不該被截"


def test_clip_std_bounds_exclude_current_period():
    """回歸測試：窗口若含當期，離群值會撐高自己的上界而截不到。"""
    rng = np.random.default_rng(5)
    vals = list(rng.normal(10.0, 0.2, 15)) + [1000.0]
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    df = pd.DataFrame({"A": vals}, index=idx)
    got = dsl.compute("clip_std(f1, 12)", {"f1": df}, {"A": "g"})["A"].iloc[-1]
    assert got < 15.0, f"含當期的寫法上界會 ~950（等於沒截），實際 {got}"


def test_clip_std_zero_variance_history_does_not_clip():
    """歷史窗口零變異時不截尾——否則會把序列硬壓成常數。"""
    vals = [10.0] * 12 + [10.5, 9.5, 10.2]
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    df = pd.DataFrame({"A": vals}, index=idx)
    got = dsl.compute("clip_std(f1, 12)", {"f1": df}, {"A": "g"})["A"]
    assert got.iloc[12] == pytest.approx(10.5), "sd=0 時應原值輸出，而非壓成 10.0"


def test_clip_std_no_lookahead():
    """未來出現巨大離群值，不能改變當期以前的截尾結果。"""
    idx = pd.period_range("2020-01", periods=30, freq="M")
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"A": rng.normal(10, 1, 30)}, index=idx)
    dirty = base.copy()
    dirty.iloc[20:] = 1e6
    gm = {"A": "g"}
    a = dsl.compute("clip_std(f1, 12)", {"f1": base}, gm).iloc[:20]
    b = dsl.compute("clip_std(f1, 12)", {"f1": dirty}, gm).iloc[:20]
    pd.testing.assert_frame_equal(a, b)


def test_streak_capped_by_window():
    """連續期數不會超過窗口 n。"""
    vals = np.arange(30, dtype=float)          # 一路單調上升
    idx = pd.period_range("2020-01", periods=len(vals), freq="M")
    df = pd.DataFrame({"A": vals}, index=idx)
    for n in (3, 6, 12):
        got = dsl.compute(f"streak(f1, {n})", {"f1": df}, {"A": "g"})["A"]
        assert got.max() == float(n), f"streak(x,{n}) 上限應為 {n}，實際 {got.max()}"


def test_streak_no_lookahead_on_reversal():
    """把未來改成劇烈反轉，當期以前的 streak 值不能變。"""
    idx = pd.period_range("2020-01", periods=12, freq="M")
    base = pd.DataFrame({"A": np.arange(12, dtype=float)}, index=idx)
    dirty = base.copy()
    dirty.iloc[8:] = -999.0
    gm = {"A": "g"}
    a = dsl.compute("streak(f1, 6)", {"f1": base}, gm).iloc[:8]
    b = dsl.compute("streak(f1, 6)", {"f1": dirty}, gm).iloc[:8]
    pd.testing.assert_frame_equal(a, b)
