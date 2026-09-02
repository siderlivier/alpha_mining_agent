"""
Stage 4b 產業專屬入庫通道測試。

方法：構造合成因子——在「金融」產業內用 fwd_ret 本身（完美預測，ICIR 極高），
其他產業填隨機雜訊。斷言只有金融通過產業門檻，且入庫後帶正確標籤。
（需 data/monthly_base.parquet，無資料時自動跳過。）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "monthly_base.parquet").exists(),
    reason="monthly_base 未建置")

TARGET = "金融"


@pytest.fixture(scope="module")
def ctx():
    import eval_candidates as ec
    c = ec.Context()
    c.lib_values, c.lib_meta = None, {}
    return c


@pytest.fixture(scope="module")
def synthetic_fac(ctx):
    """金融欄 = fwd_ret（完美預測）；其他欄 = 純雜訊。"""
    import eval_candidates as ec
    rng = np.random.default_rng(42)
    fac = pd.DataFrame(rng.normal(size=ctx.fwd.shape),
                       index=ctx.fwd.index, columns=ctx.fwd.columns)
    cols = ec.group_cols(ctx, fac)[TARGET]
    fac[cols] = ctx.fwd[cols]
    return fac


def test_industry_qualify_only_target(ctx, synthetic_fac):
    import eval_candidates as ec
    months = ctx.mask("sub_train") + ctx.mask("validation")
    ind = ec.industry_ic_series(ctx, synthetic_fac, months)
    quals = ec.industry_qualify(ctx, synthetic_fac, ind)
    names = [q[0] for q in quals]
    assert TARGET in names, "完美預測產業未通過產業門檻"
    assert names == [TARGET], f"雜訊產業不應通過: {names}"
    g, itr, iva, dec = quals[0]
    assert itr > 1.0 and iva > 1.0            # 完美預測 → ICIR 極高


def test_noise_factor_qualifies_nowhere(ctx):
    import eval_candidates as ec
    rng = np.random.default_rng(7)
    fac = pd.DataFrame(rng.normal(size=ctx.fwd.shape),
                       index=ctx.fwd.index, columns=ctx.fwd.columns)
    months = ctx.mask("sub_train") + ctx.mask("validation")
    quals = ec.industry_qualify(ctx, fac, ec.industry_ic_series(ctx, fac, months))
    assert quals == []


def test_admit_with_scope(tmp_path):
    import dsl
    from memory import Memory
    mem = Memory(root=tmp_path / "memory")
    mem.ensure()
    pf = dsl.parse("cs_rank(roe) * sign(mom_60)")
    idx = pd.period_range("2015-01", periods=4, freq="M").astype(str)
    fac = pd.DataFrame(np.zeros((4, 2)), index=idx, columns=["A", "B"])
    diag = {"verdict": "passed_industry",
            "sub_train": {"icir": 0.3}, "validation": {"icir": 0.2},
            "industry_metrics": {"train_icir": 0.7, "valid_icir": 0.4,
                                 "decay_pct": 43},
            "coverage": 0.7, "turnover_m": 0.2}
    cand = {"id": "X-1", "formula": "cs_rank(roe) * sign(mom_60)",
            "category": "fin_special", "name_zh": "金融測試", "desc_zh": "測試"}
    fid = mem.admit(cand, diag, pf, fac, industry_scope=TARGET,
                    test_metrics_sealed={"icir": 0.5})
    meta = mem.library()[fid]
    assert meta["industry_scope"] == TARGET
    assert meta["industry_metrics"]["train_icir"] == 0.7
    assert f"{TARGET}限定" in mem.library_summary()


def test_full_pool_rejects_industry_verdict_without_scope(tmp_path):
    """verdict=rejected 的候選仍然不可入庫。"""
    import dsl
    from memory import Memory
    mem = Memory(root=tmp_path / "memory")
    mem.ensure()
    pf = dsl.parse("cs_rank(roe)")
    idx = pd.period_range("2015-01", periods=3, freq="M").astype(str)
    fac = pd.DataFrame(np.zeros((3, 2)), index=idx, columns=["A", "B"])
    with pytest.raises(ValueError):
        mem.admit({"id": "X", "formula": "cs_rank(roe)", "category": "c",
                   "name_zh": "n", "desc_zh": "d"},
                  {"verdict": "rejected_stage4"}, pf, fac)
