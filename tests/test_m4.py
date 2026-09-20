"""
M4 測試：聚合審計洩漏控制、報表、整理回合安全機制、運算子提案驗證。
全部使用 tmp 目錄，絕不觸碰真實 memory/。
"""
import json

import numpy as np
import pandas as pd
import pytest

import dsl
from memory import Memory


@pytest.fixture
def mem(tmp_path):
    m = Memory(root=tmp_path / "memory")
    m.ensure()
    return m


def _admit_fake(mem, i, category, formula, tr, te):
    pf = dsl.parse(formula)
    idx = pd.period_range("2015-01", periods=4, freq="M").astype(str)
    fac = pd.DataFrame(np.random.default_rng(i).normal(size=(4, 3)),
                       index=idx, columns=["S1", "S2", "S3"])
    diag = {"verdict": "passed", "sub_train": {"mean_ic": 0.03, "icir": tr},
            "validation": {"icir": tr * 0.8, "decay_pct": 20},
            "coverage": 0.8, "turnover_m": 0.3}
    cand = {"id": f"X-{i}", "formula": formula, "category": category,
            "name_zh": f"測試{i}", "desc_zh": "測試因子"}
    mem.admit(cand, diag, pf, fac,
              test_metrics_sealed={"icir": te, "mean_ic": 0.01,
                                   "decay_vs_subtrain_pct": (1 - te / tr) * 100})


# ---- audit ----------------------------------------------------------------

def test_audit_insufficient_sample(mem):
    from audit import audit_text
    _admit_fake(mem, 1, "rev", "cs_rank(rev_yoy)", 0.5, 0.2)
    txt = audit_text(mem)
    assert "樣本不足" in txt
    assert "0.2" not in txt          # 不得反推個體 test 數字


def test_audit_aggregates_no_individual_leak(mem):
    from audit import audit_text
    _admit_fake(mem, 1, "rev", "cs_rank(rev_yoy)", 0.50, 0.20)
    _admit_fake(mem, 2, "rev", "cs_rank(rev_mom)", 0.60, 0.30)
    _admit_fake(mem, 3, "mom", "cs_rank(mom_60)", 0.50, 0.40)
    _admit_fake(mem, 4, "mom", "cs_rank(mom_120)", 0.40, 0.35)
    txt = audit_text(mem)
    assert "[audit]" in txt and "category" in txt
    for leak in ("F-001", "F-002", "F-003", "F-004"):
        assert leak not in txt, f"審計輸出洩漏個體因子 {leak}"


def test_audit_feedback_unchanged_when_sealed_test_changes(mem):
    from audit import audit_text
    import consolidate
    _admit_fake(mem, 1, "rev", "cs_rank(rev_yoy)", .5, .2)
    _admit_fake(mem, 2, "rev", "cs_rank(rev_mom)", .6, .3)
    before=audit_text(mem)
    prompts=[]
    def capture(prompt):
        prompts.append(prompt)
        return "{}"
    consolidate.run(mem=mem,mock_fn=capture)
    lib=mem.library()
    for meta in lib.values():meta["test_metrics_sealed"]={"icir":-9999}
    mem._save_library(lib)
    assert audit_text(mem)==before
    assert "validation" in before and "test" not in before
    consolidate.run(mem=mem,mock_fn=capture)
    assert prompts[0]==prompts[1]


def test_scoped_metric_pair_does_not_mix_market_train():
    from memory import admission_metrics
    m={"industry_scope":"電子","sub_train":{"icir":.2},
       "validation":{"icir":.1},"industry_metrics":{"train_icir":.8,"valid_icir":.6},
       "test_metrics_sealed":{"icir":.4,"decay_vs_subtrain_pct":-100}}
    tr,va,te=admission_metrics(m)
    assert tr["icir"]==.8 and va["decay_pct"]==pytest.approx(25) and te["decay_vs_subtrain_pct"]==50
    m.pop("industry_metrics")
    assert admission_metrics(m)[2]["decay_vs_subtrain_pct"] is None


def test_report_industry_pass_tokens_and_empty(mem):
    from report import _collect,build_html,build_md,build_text
    assert "0%" in build_html(_collect(mem))
    for verdict in ["passed","passed_industry","rejected_stage1"]:
        mem.write_attempt({"category":"x","hypothesis":"h","prediction":"p",
                           "formula":"cs_rank(roe)","verdict":verdict})
    (mem.root/"budget.json").write_text(json.dumps({"tokens_used":12345,"est_tokens_used":2}),encoding="utf-8")
    d=_collect(mem)
    assert d["by_cat"]["x"]==[3,2]
    for renderer in [build_text,build_html,build_md]:
        text=renderer(d)
        assert "67%" in text and "12,345" in text


def test_legacy_test_audit_learning_excluded_without_deleting_history(mem):
    text="## 全域規則\n1. useful\n2. [audit] secret test numbers\n   continuation\n3. [audit:validation] allowed\n"
    mem.write_learnings(text)
    clean=mem.prompt_learnings()
    assert "secret" not in clean and "continuation" not in clean
    assert "useful" in clean and "allowed" in clean
    assert mem.read_learnings()==text


# ---- report ---------------------------------------------------------------

def test_report_contains_sealed_for_human(mem):
    from report import _collect, build_html, build_text
    _admit_fake(mem, 1, "rev", "cs_rank(rev_yoy)", 0.5, 0.23)
    mem.write_attempt({"category": "rev", "hypothesis": "h", "prediction": "p",
                       "formula": "cs_rank(rev_yoy)", "verdict": "passed",
                       "failure_type": "none", "round": 1})
    d = _collect(mem)
    txt = build_text(d)
    assert "0.23" in txt             # 人類報表要能看到密封指標
    assert "勿餵 agent" in txt
    assert "漏斗統計" in txt
    h = build_html(d)
    assert "0.23" in h and "test" in h


# ---- 運算子提案驗證 --------------------------------------------------------

def _base_proposal():
    return {"name": "ts_argmax", "signature": "(x, n)",
            "semantics": "過去 n 月最大值出現位置距今月數",
            "pseudocode": "n-1-argmax(window)", "evidence": ["A-0001", "A-0002"],
            "example": "cs_rank(ts_argmax(rev_yoy, 12))", "why": "無法表達時點資訊"}


def test_proposal_validation(mem):
    from consolidate import validate_proposal
    for i in (1, 2):
        mem.write_attempt({"category": "x", "hypothesis": "h", "prediction": "p",
                           "formula": "f", "verdict": "rejected_stage1",
                           "failure_type": "expression_bad"})
    assert validate_proposal(_base_proposal(), mem) is None
    # 名稱不合法
    assert "名稱" in validate_proposal({**_base_proposal(), "name": "TS-Max!"}, mem)
    # 帶常數參數的簽名 → 拒
    assert "簽名" in validate_proposal(
        {**_base_proposal(), "signature": "(x, n, threshold)"}, mem)
    # 證據不足
    assert "證據" in validate_proposal(
        {**_base_proposal(), "evidence": ["A-0001"]}, mem)
    # 引用不存在的 attempt
    assert "不存在" in validate_proposal(
        {**_base_proposal(), "evidence": ["A-0001", "A-9999"]}, mem)


# ---- 整理回合安全機制 ------------------------------------------------------

def test_consolidate_rejects_bad_rewrite(mem, monkeypatch):
    import consolidate
    mem.append_learning("全域規則", "- [G-01] 原有規則。")
    original = mem.read_learnings()

    def bad_llm(prompt):
        return json.dumps({"learnings_md": "只有一句話沒有小節",
                           "operator_proposals": [], "notes": "壞輸出"})
    res = consolidate.run(mem=mem, mock_fn=bad_llm)
    assert res["learnings_updated"] is False
    assert mem.read_learnings() == original     # 原文保留


def test_consolidate_backup_and_rewrite(mem):
    import consolidate
    mem.append_learning("全域規則", "- [G-01] 舊規則。")
    new_md = ("# 因子挖掘經驗庫\n\n## 全域規則\n\n- [G-01|promoted] 新規則。\n\n"
              "## 禁忌方向\n\n（尚無）\n\n## 待驗證假設佇列\n\n（尚無）\n")

    def good_llm(prompt):
        return json.dumps({"learnings_md": new_md,
                           "operator_proposals": [_base_proposal()],
                           "notes": "ok"}, ensure_ascii=False)
    for i in (1, 2):
        mem.write_attempt({"category": "x", "hypothesis": "h", "prediction": "p",
                           "formula": "f", "verdict": "rejected_stage1",
                           "failure_type": "expression_bad"})
    res = consolidate.run(mem=mem, mock_fn=good_llm)
    assert res["learnings_updated"] and res["proposals"] == 1
    assert "promoted" in mem.read_learnings()
    backups = list((mem.root / "learnings_history").glob("*.md"))
    assert len(backups) == 1 and "G-01] 舊規則" in backups[0].read_text(encoding="utf-8")
    pp = json.loads((mem.root / "operator_proposals.json").read_text(encoding="utf-8"))
    assert pp["pending"][0]["name"] == "ts_argmax"
    assert pp["pending"][0]["status"] == "pending_review"
