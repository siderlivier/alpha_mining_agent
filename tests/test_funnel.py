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

pytestmark = pytest.mark.skipif(not HAS_DATA, reason="monthly_base 未建置")


@pytest.fixture(scope="module")
def ctx():
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
