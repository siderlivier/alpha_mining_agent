"""
參考因子匯入的篩選邏輯測試——重點在「彼此去重複」這一道。

為什麼要有這支：原本的三道篩選（t 值、同向、ICIR、衰減）全是**逐因子**的，
看不到「兩個因子其實是同一條公式」。前置專案 `mine_dfs.py` 就有這種情況：
`net_margin` 與 `ni_to_rev` 都是稅後淨利 ÷ 營收，ρ = 1.000；
`op_margin` 與 `op_to_rev` 同理。兩個都匯入的後果是因子數灌水，
而且 factor_lab 等權合成時「淨利率」這個概念被賦予兩倍權重。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import seed_reference as sr        # noqa: E402

STOCKS = [f"S{i:03d}" for i in range(30)]
YM = [str(p) for p in pd.period_range("2015-01", periods=36, freq="M")]


def _long(spec: dict[str, np.ndarray]) -> pd.DataFrame:
    """spec: {因子名: 形狀 (月, 股) 的值}，轉成長格式。"""
    frames = []
    for name, arr in spec.items():
        for i, m in enumerate(YM):
            frames.append(pd.DataFrame({"dfs_name": name, "ym": m,
                                        "stock_id": STOCKS, "value": arr[i]}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(len(YM), len(STOCKS)))
    b = rng.normal(size=(len(YM), len(STOCKS)))
    spec = {
        "orig":       a,
        "exact_dup":  a.copy(),            # 完全一樣，ρ = 1.000
        "scaled_dup": a * 3.0 + 7.0,       # 單調變換，排名相同 → ρ = 1.000
        "near_dup":   a + 0.05 * b,        # 高度相關但不完全一樣
        "unrelated":  b,                   # 獨立
    }
    row = pd.DataFrame(
        {"ICIR_train": [0.60, 0.55, 0.50, 0.45, 0.40]},
        index=["orig", "exact_dup", "scaled_dup", "near_dup", "unrelated"])
    return _long(spec), list(spec), row


def test_drops_exact_duplicates(sample):
    long, names, row = sample
    keep, dropped = sr.dedupe_reference(long, names, row, max_corr=0.95)
    assert "orig" in keep, "|ICIR_train| 最大的應該被保留"
    assert "unrelated" in keep, "獨立因子不該被丟"
    dropped_names = {d[0] for d in dropped}
    assert {"exact_dup", "scaled_dup"} <= dropped_names


def test_keeps_the_stronger_of_a_duplicate_pair(sample):
    """重複的一對裡，保留 |ICIR_train| 較大的那個。"""
    long, names, row = sample
    row = row.copy()
    row.loc["exact_dup", "ICIR_train"] = 0.99      # 讓副本變成比較強的那個
    keep, _ = sr.dedupe_reference(long, names, row, max_corr=0.95)
    assert "exact_dup" in keep and "orig" not in keep


def test_reports_who_it_collided_with(sample):
    """回報要說清楚是「跟誰」重複、ρ 多少——不然人工沒法判斷該不該接受。"""
    long, names, row = sample
    _, dropped = sr.dedupe_reference(long, names, row, max_corr=0.95)
    for name, against, rho in dropped:
        assert name in names and against in names
        assert 0.95 < rho <= 1.0 + 1e-9


def test_threshold_one_disables_dedup(sample):
    """--max-ref-corr 1.0 = 完全不去重（保留舊行為的逃生口）。"""
    long, names, row = sample
    keep, dropped = sr.dedupe_reference(long, names, row, max_corr=1.0)
    assert not dropped and len(keep) == len(names)


def test_preserves_original_order(sample):
    """保留的名稱要維持原本的順序，R-xxx 編號才不會每次匯入都跳動。"""
    long, names, row = sample
    keep, _ = sr.dedupe_reference(long, names, row, max_corr=0.95)
    assert keep == [n for n in names if n in keep]


def test_correlation_is_rank_based_within_month(sample):
    """
    去重用的是「月內排名」相關，不是原始值的 Pearson。

    單調變換（×3 +7）不改變任何一個月的排名，所以必須被判為重複。
    用原始值的 Pearson 也會抓到這一例，但遇到非線性單調變換就會漏掉——
    排名法才與 Stage 2 的口徑一致。
    """
    long, names, row = sample
    # 加一個非線性但保序的變換：exp 之後排名不變
    extra = long[long["dfs_name"] == "orig"].copy()
    extra["dfs_name"] = "mono_dup"
    extra["value"] = np.exp(extra["value"])
    long2 = pd.concat([long, extra], ignore_index=True)
    names2 = names + ["mono_dup"]
    row2 = pd.concat([row, pd.DataFrame({"ICIR_train": [0.35]},
                                        index=["mono_dup"])])
    keep, dropped = sr.dedupe_reference(long2, names2, row2, max_corr=0.95)
    assert "mono_dup" in {d[0] for d in dropped}, \
        "保序的非線性變換沒被抓到——相關度可能誤用了原始值而非排名"
