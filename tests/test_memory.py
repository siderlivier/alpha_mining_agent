"""
記憶層單元測試（tmp 目錄，不觸碰真實 memory/）。
"""
import json

import numpy as np
import pandas as pd
import pytest

import dsl
from memory import Memory, derive_aspect


@pytest.fixture
def mem(tmp_path):
    m = Memory(root=tmp_path / "memory")
    m.ensure()
    return m


def _fake_factor():
    pf = dsl.parse("cs_rank(rev_yoy_ma3) * sign(mom_60)")
    idx = pd.period_range("2015-01", periods=6, freq="M").astype(str)
    fac = pd.DataFrame(np.random.default_rng(0).normal(size=(6, 3)),
                       index=idx, columns=["S1", "S2", "S3"])
    return pf, fac


PASSED_DIAG = {"verdict": "passed",
               "sub_train": {"mean_ic": 0.03, "icir": 0.5},
               "validation": {"mean_ic": 0.02, "icir": 0.4, "decay_pct": 20},
               "coverage": 0.8, "turnover_m": 0.3}
CAND = {"id": "X-01", "formula": "cs_rank(rev_yoy_ma3) * sign(mom_60)",
        "category": "revenue_momentum",
        "name_zh": "營收動能價格確認", "desc_zh": "營收年增排名，僅在動能為正時做多"}


# ---- attempts --------------------------------------------------------------

def test_attempt_id_sequence(mem):
    a = {"category": "x", "hypothesis": "h", "prediction": "p",
         "formula": "f", "verdict": "rejected_stage1"}
    assert mem.write_attempt(dict(a)) == "A-0001"
    assert mem.write_attempt(dict(a)) == "A-0002"


def test_attempt_required_fields(mem):
    with pytest.raises(ValueError, match="hypothesis"):
        mem.write_attempt({"category": "x", "prediction": "p",
                           "formula": "f", "verdict": "v"})


def test_attempt_immutable(mem):
    a = {"id": "A-0001", "category": "x", "hypothesis": "h",
         "prediction": "p", "formula": "f", "verdict": "v"}
    mem.write_attempt(dict(a))
    with pytest.raises(FileExistsError):
        mem.write_attempt(dict(a))   # 只進不出：同 id 禁止覆寫


# ---- 面向推導 --------------------------------------------------------------

def test_derive_aspect():
    fmap = {"mom_60": "技術面", "vol_63": "技術面",
            "roe": "基本面", "rev_yoy_ma3": "基本面"}
    assert derive_aspect({"mom_60", "vol_63"}, fmap) == "技術面"
    assert derive_aspect({"roe"}, fmap) == "基本面"
    assert derive_aspect({"mom_60", "roe"}, fmap) == "混合"


# ---- 因子庫 ----------------------------------------------------------------

def test_admit_requires_zh_metadata(mem):
    pf, fac = _fake_factor()
    bad = {k: v for k, v in CAND.items() if k != "name_zh"}
    with pytest.raises(ValueError, match="name_zh"):
        mem.admit(bad, PASSED_DIAG, pf, fac)


def test_admit_rejects_unpassed(mem):
    pf, fac = _fake_factor()
    with pytest.raises(ValueError, match="passed"):
        mem.admit(CAND, {**PASSED_DIAG, "verdict": "rejected_stage4"}, pf, fac)


def test_admit_and_metadata(mem):
    pf, fac = _fake_factor()
    fid = mem.admit(CAND, PASSED_DIAG, pf, fac,
                    test_metrics_sealed={"icir": 0.42}, round_id=7)
    meta = mem.library()[fid]
    assert fid == "F-001"
    assert meta["name_zh"] == "營收動能價格確認"
    assert meta["desc_zh"]
    assert meta["aspect"] == "混合"          # rev_yoy_ma3(基本面) × mom_60(技術面)
    assert meta["created_at"]
    assert meta["round"] == 7
    assert meta["test_metrics_sealed"] == {"icir": 0.42}
    # 因子值落盤
    fv = pd.read_parquet(mem.values_path)
    assert set(fv["factor_id"]) == {"F-001"}


def test_admit_blocks_equivalent_formula(mem):
    pf, fac = _fake_factor()
    mem.admit(CAND, PASSED_DIAG, pf, fac)
    pf2 = dsl.parse("sign(mom_60) * cs_rank(rev_yoy_ma3)")   # 交換律等價
    with pytest.raises(ValueError, match="等價"):
        mem.admit({**CAND, "id": "X-02",
                   "formula": "sign(mom_60) * cs_rank(rev_yoy_ma3)"},
                  PASSED_DIAG, pf2, fac)


def test_library_summary_no_leak(mem):
    """摘要必須含中文名/面向，且不得含任何績效數字與密封 test 指標。"""
    pf, fac = _fake_factor()
    mem.admit(CAND, PASSED_DIAG, pf, fac,
              test_metrics_sealed={"icir": 0.9876, "mean_ic": 0.0654})
    s = mem.library_summary()
    assert "營收動能價格確認" in s and "混合" in s
    for leak in ("0.9876", "0.0654", "icir", "0.5", "sealed"):
        assert leak not in s, f"library_summary 洩漏: {leak}"


# ---- 經驗檔 ----------------------------------------------------------------

def test_learnings_skeleton_and_append(mem):
    text = mem.read_learnings()
    assert "## 全域規則" in text and "（尚無）" in text
    mem.append_learning("全域規則", "- [G-01|single_case] 測試規則。證據：A-0001。")
    text = mem.read_learnings()
    assert "G-01" in text
    # 該節占位移除，其他節保留
    head = text.split("## 禁忌方向")[0]
    assert "（尚無）" not in head
    assert "## 禁忌方向" in text and "## 待驗證假設佇列" in text


def test_learnings_append_new_section(mem):
    mem.append_learning("revenue_momentum", "- [R-01] 新小節第一條。")
    assert "## revenue_momentum" in mem.read_learnings()
