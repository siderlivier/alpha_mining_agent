"""
全管線前瞻偏差檢測（規格書 10 章的延伸）。

test_dsl_no_lookahead.py 只測 DSL 層；這支往上測到整條評估管線：
  monthly_base → Context 載入/對齊 → 四階段漏斗 → 診斷數字

方法：造一份合成 monthly_base，複製一份把 t_cut 之後的**所有欄位**汙染成
極端值，兩邊各跑一次完整的 evaluate_batch，斷言 sub-train 期的診斷數字
逐項相同。任何一處讀到未來資料，被汙染的那份必然算出不同的 sub-train 指標。

同時檢查三件對齊面的事：
  - Stage 2 的去相關只用 sub_train 窗口（不得碰 validation/test）
  - 參考因子（R-xxx）進入 Stage 2 後同樣不得引入前瞻
  - fwd_ret_1m（label，本來就看未來）不得出現在可用欄位裡
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# 汙染起點 = sub_train 的最後一個月（從 config 讀，避免寫死而與切分脫節）。
# ⚠️ 這裡曾經寫死 index，結果汙染點落在 sub_train 內部，測試自己製造了假陽性。
import yaml as _yaml
_CFG = _yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml")
                       .read_text(encoding="utf-8"))
CUT_YM = _CFG["split"]["sub_train"][1]        # 例如 "2017-12"
N_MONTHS = 110
STOCKS = [f"S{i:03d}" for i in range(40)]
GROUPS = {s: ("半導體" if i % 4 == 0 else "電子" if i % 4 == 1
              else "生技" if i % 4 == 2 else "金融")
          for i, s in enumerate(STOCKS)}


def _months():
    return [str(p) for p in pd.period_range("2012-01", periods=N_MONTHS, freq="M")]


def _make_base(seed=7):
    """合成 monthly_base，欄位名與 fields.yaml 一致。"""
    import yaml
    fy = yaml.safe_load((ROOT / "fields.yaml").read_text(encoding="utf-8"))
    fields = [f for grp in fy.values() for f in grp]
    rng = np.random.default_rng(seed)
    ym = _months()
    idx = pd.MultiIndex.from_product([STOCKS, ym], names=["stock_id", "ym"])
    df = pd.DataFrame(index=idx).reset_index()
    df["group"] = df["stock_id"].map(GROUPS)
    n = len(df)
    for f in fields:
        v = rng.normal(size=n).cumsum() / 50 + rng.normal(size=n)
        v[rng.random(n) < 0.05] = np.nan          # 保留缺值路徑
        df[f] = v
    df["fwd_ret_1m"] = rng.normal(0, 0.08, n)
    return df, fields


def _corrupt(df, fields, cut_ym):
    """把 cut_ym 之後的所有因子欄位換成極端值（label 保持不變）。"""
    bad = df.copy()
    mask = bad["ym"] > cut_ym
    for f in fields:
        bad.loc[mask, f] = 1e7
    return bad


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """把合成資料寫進暫時的專案結構，並回傳兩份（乾淨/汙染）的路徑。"""
    base, fields = _make_base()
    ym = _months()
    cut = CUT_YM
    assert cut in ym, f"CUT_YM {cut} 不在合成月份範圍內"
    assert ym[-1] > cut, "合成資料必須涵蓋 cut 之後的月份，否則沒東西可汙染"
    d = tmp_path_factory.mktemp("pipe")
    (d / "data").mkdir()
    base.to_parquet(d / "data" / "clean.parquet", index=False)
    _corrupt(base, fields, cut).to_parquet(d / "data" / "dirty.parquet", index=False)
    # regime 與大盤月報酬（Context 需要）
    (d / "data" / "regime_table.json").write_text(json.dumps(
        {y: {"regime": "盤整", "taiex_ret": 0.0, "note": ""}
         for y in sorted({m[:4] for m in ym})}, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"ym": ym, "taiex_ret": np.linspace(-0.05, 0.05, len(ym))}) \
        .to_parquet(d / "data" / "market_monthly.parquet", index=False)
    return d, fields, cut


def _run(monkeypatch, d, which, extra_values=None):
    """用指定的 monthly_base 跑一次完整評估，回傳 {候選id: 診斷}。"""
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    import eval_candidates as ec
    importlib.reload(ec)

    monkeypatch.setitem(ec.CFG["paths"], "monthly_base", f"data/{which}.parquet")
    monkeypatch.setitem(ec.CFG["paths"], "regime_table", "data/regime_table.json")
    monkeypatch.setitem(ec.CFG["paths"], "memory_dir", "memory")
    monkeypatch.setattr(ec, "ROOT", d)
    (d / "memory").mkdir(exist_ok=True)
    if extra_values is not None:
        extra_values.to_parquet(d / "memory" / "factor_values.parquet", index=False)
    elif (d / "memory" / "factor_values.parquet").exists():
        (d / "memory" / "factor_values.parquet").unlink()

    ctx = ec.Context()
    cands = _candidates()
    return {r["id"]: r for r in ec.evaluate_batch(cands, ctx)}, ctx


def _candidates():
    """覆蓋全部 33 個運算子的候選集（與前瞻測試同一批公式思路）。"""
    formulas = [
        "cs_rank(ts_mean(roe, 12))",
        "cs_rank(ts_std(gross_margin, 12))",
        "cs_rank(ts_rank(rev_yoy, 12))",
        "cs_rank(delta(op_margin, 12))",
        "cs_rank(delay(ep, 3))",
        "cs_rank(ts_slope(net_margin, 12))",
        "cs_rank(ts_corr(rev_yoy, eps_yoy, 12))",
        "cs_rank(streak(op_margin, 6))",
        "cs_rank(streak_true(greater(roe, 0), 12))",
        "cs_rank(clip_std(rev_mom, 24))",
        "rank_nz(if_else(greater(rev_yoy, 0), px_hi252, 0))",
        "cs_rank(industry_demean(op_margin))",
        "cs_z(sdiv(ep, vol_63))",
        "cs_rank_all(log1p_abs(amt_21))",
        "cs_rank(add(mul(bp, sign(roe)), neg(abs(sp))))",
        "cs_rank(sub(ts_max(roe, 24), ts_min(roe, 24)))",
        "cs_rank(ts_med(turn_21, 12))",
        "if_else(and_(greater(roe, 0), less(accruals, 0)), cs_rank(ep), cs_rank(bp))",
        "if_else(or_(greater(rev_yoy, 0), greater(eps_yoy, 0)), cs_rank(sp), cs_rank(ocf_ratio))",
        "cs_rank(ts_resi(div_yield, 24))",
        "cs_rank(ts_rsq(frgn_ratio, 12))",
    ]
    return [{"id": f"C-{i}", "category": "t", "direction": "pos",
             "hypothesis": "h", "prediction": "p", "formula": f,
             "name_zh": "測", "desc_zh": "測"}
            for i, f in enumerate(formulas, 1)]


# ---------------------------------------------------------------------------

def _subtrain_only(rec):
    """只取 sub-train 期的診斷數字（validation/test 期本來就會被汙染改變）。"""
    return {"verdict_stage1": rec.get("verdict") in ("rejected_stage1",),
            "sub_train": rec.get("sub_train"),
            "coverage_is_none": rec.get("coverage") is None}


@pytest.fixture(scope="module")
def runs(env, request):
    """乾淨/汙染各跑一次就好——整條管線很慢，別在每個測試重跑。"""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    request.addfinalizer(mp.undo)
    d, _, _ = env
    clean, ctx = _run(mp, d, "clean")
    dirty, _ = _run(mp, d, "dirty")
    return clean, dirty, ctx


def test_pipeline_no_lookahead_in_subtrain(runs):
    """核心斷言：汙染未來，sub-train 的診斷數字必須逐項不變。"""
    clean, dirty, _ = runs
    assert set(clean) == set(dirty)
    diffs = []
    for cid in clean:
        a, b = _subtrain_only(clean[cid]), _subtrain_only(dirty[cid])
        if a != b:
            diffs.append((cid, clean[cid].get("formula", ""), a, b))
    assert not diffs, "sub-train 診斷被未來資料改變了：\n" + "\n".join(
        f"  {c}: {f}\n    clean={a}\n    dirty={b}" for c, f, a, b in diffs)


def test_subtrain_window_ends_before_validation(runs):
    """切分不得重疊：sub_train 的最後一個月必須早於 validation 的第一個月。"""
    _, _, ctx = runs
    tr, va, te = (ctx.mask("sub_train"), ctx.mask("validation"), ctx.mask("test"))
    assert tr and va and te
    assert max(tr) < min(va) < max(va) < min(te), (max(tr), min(va), max(va), min(te))
    assert not (set(tr) & set(va)) and not (set(va) & set(te))


def test_label_not_exposed_as_field(runs):
    """fwd_ret_1m 是 label（本來就看未來），絕不可出現在可用欄位裡。"""
    _, _, ctx = runs
    assert "fwd_ret_1m" not in ctx.fields
    assert "fwd_ret_1m" not in ctx.data


def test_reference_factors_do_not_leak_future(env, runs, monkeypatch):
    """
    參考因子進入 Stage 2 後同樣不得引入前瞻：
    造一個「t_cut 之後全是極端值」的參考因子，sub-train 診斷仍須不變。
    """
    d, _, cut = env
    clean, _, _ = runs

    ym = _months()
    rng = np.random.default_rng(3)
    rows = []
    for m in ym:
        v = rng.normal(size=len(STOCKS))
        if m > cut:
            v[:] = 1e7                      # 未來全汙染
        rows.append(pd.DataFrame({"factor_id": "R-001", "ym": m,
                                  "stock_id": STOCKS, "value": v}))
    ref = pd.concat(rows, ignore_index=True)

    withref, ctx = _run(monkeypatch, d, "clean", extra_values=ref)
    assert ctx.lib_values and "R-001" in ctx.lib_values, "參考因子沒被 Stage 2 載入"
    diffs = [cid for cid in clean
             if _subtrain_only(clean[cid])["sub_train"]
             != _subtrain_only(withref[cid])["sub_train"]]
    assert not diffs, f"參考因子改變了 sub-train 診斷（不該發生）：{diffs}"
