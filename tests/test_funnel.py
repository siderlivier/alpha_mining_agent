"""
四階段漏斗迴歸測試（需 data/monthly_base.parquet，無資料時自動跳過）。

驗證 M1 驗收時確立的行為快照：
  - 語法錯誤 → rejected_syntax
  - 低波動因子 → 死於 Stage 4 多頭腿檢查（與 FinLab 文章結論一致）
  - 營收動能複合因子 → passed
  - 診斷 JSON 含全部八類素材
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / "data" / "monthly_base.parquet").exists()

@pytest.fixture(scope="module")
def ctx():
    if not HAS_DATA:
        pytest.skip("monthly_base 未建置")
    import eval_candidates as ec
    c = ec.Context()
    # 隔離：清空因子庫快取，讓測試結果不隨真實 memory/ 的庫內容漂移
    c.lib_values, c.lib_meta = None, {}
    return c


def _run(ctx, batch):
    import eval_candidates as ec
    return {r["id"]: r for r in ec.evaluate_batch(batch, ctx)}


def test_funnel_verdicts(ctx):
    batch = [
        {"id": "T-03", "formula": "ts_mean(f_bad, 6)",
         "category": "x", "direction": "pos", "hypothesis": "x", "prediction": "x"},
        {"id": "T-05", "formula": "cs_rank(vol_126)",
         "category": "volatility", "direction": "pos",
         "hypothesis": "宣稱高波動有正報酬（實際 IC 為負，應觸發方向檢查）",
         "prediction": "ICIR>0.3"},
        {"id": "T-04", "formula": "cs_rank(rev_yoy_ma3 + rev_yoy)",
         "category": "revenue_momentum", "direction": "pos",
         "hypothesis": "營收動能複合", "prediction": "ICIR>0.4"},
    ]
    res = _run(ctx, batch)
    assert res["T-03"]["verdict"] == "rejected_syntax"
    # 台股波動因子 IC 為負：宣稱 pos 必觸發 pre-registration 方向檢查
    assert res["T-05"]["verdict"] == "rejected_stage1"
    assert "方向" in res["T-05"]["reason"]
    assert res["T-04"]["verdict"] == "passed"


def test_diagnostics_complete(ctx):
    batch = [{"id": "D-01", "formula": "cs_rank(rev_yoy_ma3 + rev_yoy)",
              "category": "revenue_momentum", "direction": "pos",
              "hypothesis": "x", "prediction": "x"}]
    r = _run(ctx, batch)["D-01"]
    for key in ("sub_train", "validation", "ic_by_year", "cond_ic",
                "industry_icir", "legs", "coverage", "turnover_m"):
        assert key in r, f"診斷缺少 {key}"
    # 逐年 IC 必須帶 regime 標籤
    year, (ic, regime) = next(iter(r["ic_by_year"].items()))
    assert regime in ("多頭", "空頭", "盤整", "資料不足", "?")
    # ⛔ 洩漏檢查：診斷輸出不得含 test 期年份（2020+）
    assert all(int(y) < 2020 for y in r["ic_by_year"]), "test 期資訊洩漏！"


def test_no_test_period_leak_in_output(ctx):
    """任何 verdict 的輸出 JSON 序列化後都不得出現 test 期指標。"""
    import json
    batch = [{"id": "L-01", "formula": "cs_rank(mom_120) * sign(rev_yoy_ma3)",
              "category": "momentum", "direction": "pos",
              "hypothesis": "x", "prediction": "x"}]
    txt = json.dumps(_run(ctx, batch), ensure_ascii=False)
    for y in range(2020, 2027):
        assert f'"{y}"' not in txt, f"輸出含 test 期年份 {y}"


@pytest.fixture
def synthetic_context():
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(61)
    months = pd.period_range("2012-01", periods=96, freq="M").astype(str)
    cols = [f"S{i:02d}" for i in range(40)]
    values = pd.DataFrame(np.tile(np.arange(40), (96, 1)), index=months, columns=cols)
    ret = -values * 0.002 + rng.normal(0, 0.06, values.shape)
    spans = {"sub_train": list(months[:72]), "validation": list(months[72:]), "test": []}
    return SimpleNamespace(fields={"x"}, data={"x": values},
        fwd=ret, months=list(months), group_map=dict.fromkeys(cols, "A"),
        lib_values=None, lib_meta={}, regime={}, mkt=pd.Series(dtype=float),
        mask=lambda span: spans[span])


def test_negative_direction_end_to_end(synthetic_context, tmp_path):
    import pandas as pd
    import eval_candidates as ec
    from admit import admit_from_results
    from memory import Memory
    c = {"id": "negative", "formula": "cs_rank(x)", "direction": "neg",
         "name_zh": "負向測試", "desc_zh": "高值低報酬", "category": "test"}
    diag = ec.evaluate_batch([c], synthetic_context)[0]
    assert diag["verdict"] == "passed"
    assert diag["raw_mean_ic"] < 0 < diag["sub_train"]["mean_ic"]
    assert diag["value_orientation"] == -1
    assert diag["turnover_m"] == 0
    mem = Memory(tmp_path / "memory")
    [fid] = admit_from_results([c], [diag], synthetic_context, mem)
    meta = mem.library()[fid]
    assert meta["direction"] == "neg" and meta["value_orientation"] == -1
    stored = pd.read_parquet(mem.values_path)
    assert stored["value"].max() <= 0
    # 原公式相同但宣告錯誤，不能用自動翻向掩蓋假設失敗。
    wrong = dict(c, direction="pos")
    assert ec.evaluate_batch([wrong], synthetic_context)[0]["verdict"] == "rejected_stage1"


def _decision_diag():
    return {"sub_train": {"mean_ic": 0.1, "icir": 0.8},
        "validation": {"mean_ic": 0.08, "icir": 0.6, "decay_pct": 25.0},
        "ic_by_year": {}, "cond_ic": {}, "industry_icir": {},
        "legs": {"long_excess_ann": 0.1, "short_excess_ann": 0.1,
                 "short_turnover_m": 0.1},
        "coverage": 0.9, "turnover_m": 0.0, "_ind_series": {}}


@pytest.mark.parametrize("field,value", [
    ("mean_ic", 0.0196), ("icir", 0.446), ("valid_icir", 0.296),
    ("decay", 50.4), ("coverage", 0.596), ("turnover", 0.404)])
def test_thresholds_use_raw_values(monkeypatch, synthetic_context, field, value):
    import eval_candidates as ec
    diag = _decision_diag()
    if field in ("mean_ic", "icir"):
        diag["sub_train"][field] = value
    elif field == "valid_icir":
        diag["validation"]["icir"] = value
    elif field == "decay":
        diag["validation"]["decay_pct"] = value
    else:
        diag["turnover_m" if field == "turnover" else field] = value
    monkeypatch.setattr(ec, "diagnostics", lambda *a: diag)
    monkeypatch.setattr(ec, "industry_qualify", lambda *a: [])
    result = ec.evaluate_batch([{"id": "X", "formula": "cs_rank(x)"}], synthetic_context)[0]
    assert result["verdict"].startswith("rejected_stage")


def test_diagnostics_preserve_precision(monkeypatch, synthetic_context):
    import eval_candidates as ec
    monkeypatch.setattr(ec, "icir", lambda *a: (0.0196, 0.446))
    monkeypatch.setattr(ec, "coverage", lambda *a: 0.596)
    monkeypatch.setattr(ec, "leg_stats", lambda *a: (
        {"long_excess_ann": 0.00004, "short_excess_ann": -0.2}, 0.404))
    d = ec.diagnostics(synthetic_context, synthetic_context.data["x"])
    assert d["sub_train"] == {"mean_ic": 0.0196, "icir": 0.446}
    assert d["coverage"] == 0.596 and d["turnover_m"] == 0.404
    assert d["legs"]["long_excess_ann"] == 0.00004


@pytest.mark.parametrize("short_turn,verdict", [(0.0, "passed"), (0.404, "rejected_stage4"), (None, "rejected_stage4")])
def test_short_only_uses_short_turnover(monkeypatch, synthetic_context, short_turn, verdict):
    import eval_candidates as ec
    d = _decision_diag()
    d["legs"].update(long_excess_ann=-0.01, short_excess_ann=0.1,
                     short_turnover_m=short_turn)
    monkeypatch.setattr(ec, "diagnostics", lambda *a: d)
    monkeypatch.setattr(ec, "industry_qualify", lambda *a: [])
    monkeypatch.setitem(ec.FUN["stage4"], "allow_short_only", True)
    result = ec.evaluate_batch([{"id": "X", "formula": "cs_rank(x)"}], synthetic_context)[0]
    assert result["verdict"] == verdict
    assert result["trading_use"] == "short_only"
    assert result["turnover_m"] == short_turn


def test_duplicate_ids_fail_before_evaluation(synthetic_context):
    import eval_candidates as ec
    with pytest.raises(ValueError, match="唯一"):
        ec.evaluate_batch([{"id": "X", "formula": "cs_rank(x)"},
                           {"id": "X", "formula": "bad(x)"}], synthetic_context)


def test_short_only_admission_marks_usage(monkeypatch, synthetic_context, tmp_path):
    import eval_candidates as ec
    from admit import admit_from_results
    from memory import Memory
    d = _decision_diag()
    d["legs"].update(long_excess_ann=-0.01, short_excess_ann=0.1)
    monkeypatch.setattr(ec, "diagnostics", lambda *a: d)
    monkeypatch.setitem(ec.FUN["stage4"], "allow_short_only", True)
    cand = {"id": "S", "formula": "cs_rank(x)", "direction": "pos",
            "name_zh": "空頭測試", "desc_zh": "僅空頭腿有效"}
    diag = ec.evaluate_batch([cand], synthetic_context)[0]
    mem = Memory(tmp_path / "memory")
    [fid] = admit_from_results([cand], [diag], synthetic_context, mem)
    assert mem.library()[fid]["trading_use"] == "short_only"
    assert "short_only" in mem.library_summary()


def test_batch_dedupe_uses_unrounded_icir(monkeypatch, synthetic_context):
    import eval_candidates as ec
    low, high = _decision_diag(), _decision_diag()
    low["sub_train"]["icir"] = 0.8001
    high["sub_train"]["icir"] = 0.8002
    diags = iter([low, high])
    monkeypatch.setattr(ec, "diagnostics", lambda *a: next(diags))
    monkeypatch.setattr(ec, "xsec_corr", lambda *a, **k: 1.0)
    batch = [{"id": "low", "formula": "cs_rank(x)"},
             {"id": "high", "formula": "neg(cs_rank(x))"}]
    low_result, high_result = ec.evaluate_batch(batch, synthetic_context)
    assert low_result["verdict"] == "rejected_stage3"
    assert high_result["verdict"] == "passed"
