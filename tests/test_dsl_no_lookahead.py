"""
前瞻偏差測試——整個系統可信度的基石（規格書 10 章）。

方法：對覆蓋全部運算子的公式集，先在乾淨資料上計算基準因子值；
再把 t_cut 之後所有月份、所有欄位注入極端異常值（1e6）重算；
斷言 t_cut（含）以前的因子值逐格完全一致（NaN 位置也一致）。

任何運算子若「向前看」，未來的異常值必然汙染歷史值，測試即失敗。
"""
import numpy as np
import pandas as pd
import pytest

import dsl
from conftest import make_data, GROUP_MAP

T_CUT = 30  # 異常值注入起點（index 位置）

# 覆蓋全部運算子的公式集。
# ⚠️ 新增運算子時必須同步在這裡加一條用到它的公式，
#    否則下方 test_all_operators_covered 會失敗（這是刻意的守門測試）。
FORMULAS = [
    # 2026-08-27 人工核准的兩個提案
    "streak_true(greater(f1, 0), 12)",
    "streak_true(and_(f1 > 0, f2 > 0), 6)",
    "cs_rank(streak_true(greater(f1, 0), 6))",
    "clip_std(f1, 12)",
    "cs_rank(sdiv(f1, clip_std(f2, 24)))",
    # 2026-08-19 人工核准的三個提案
    "streak(f1, 6)",
    "streak(f1, 12) * sign(f2)",
    "rank_nz(if_else(greater(f1, 0), f2, 0))",
    "industry_demean(f1)",
    "cs_rank(industry_demean(f1))",
    "ts_mean(f1, 6)",
    "ts_std(f1, 12)",
    "ts_max(f1, 6)",
    "ts_min(f1, 6)",
    "ts_med(f1, 12)",
    "ts_rank(f1, 12)",
    "delay(f1, 3)",
    "delta(f1, 12)",
    "ts_slope(f1, 12)",
    "ts_rsq(f1, 12)",
    "ts_resi(f1, 12)",
    "ts_corr(f1, f2, 12)",
    "cs_rank(f1)",
    "cs_rank_all(f1)",
    "cs_z(f1)",
    "f1 + f2",
    "f1 - f2",
    "f1 * f2",
    "sdiv(f1, f2)",
    "log1p_abs(f1)",
    "abs(f1) + sign(f2)",
    "neg(f1)",
    "if_else(f1 > 0, f2, neg(f2))",
    "if_else(and_(f1 > 0, f2 > 0), f1, f2)",
    "if_else(or_(f1 < 0, f2 < 0), f1, f2)",
    "if_else(greater(f1, ts_med(f1, 24)), cs_rank(f2), neg(cs_rank(f3)))",
    "ts_rank(f1, 12) * sign(f2)",
    "cs_rank(f1) - cs_rank(delay(f1, 12))",
]


def _corrupt(data, t_cut):
    out = {}
    for k, df in data.items():
        arr = df.copy()
        arr.iloc[t_cut + 1:, :] = 1e6
        out[k] = arr
    return out


@pytest.mark.parametrize("expr", FORMULAS)
def test_no_lookahead(expr):
    clean = make_data()
    dirty = _corrupt(clean, T_CUT)
    base = dsl.compute(expr, clean, GROUP_MAP).iloc[: T_CUT + 1]
    poll = dsl.compute(expr, dirty, GROUP_MAP).iloc[: T_CUT + 1]
    pd.testing.assert_frame_equal(base, poll, check_exact=False, rtol=1e-12,
                                  obj=f"lookahead in {expr}")


def test_all_operators_covered():
    """公式集必須覆蓋白名單中的每一個運算子（防止未來加運算子卻忘了測）。"""
    covered = set()
    for expr in FORMULAS:
        pf = dsl.parse(expr)
        def walk(n):
            if n[0] == "call":
                covered.add(n[1])
                for a in n[2]:
                    walk(a)
        walk(pf.tree)
    missing = set(dsl._SIGNATURES) - covered
    assert not missing, f"未被前瞻測試覆蓋的運算子: {missing}"
