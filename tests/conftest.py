import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

N_MONTHS = 48
STOCKS = [f"S{i:02d}" for i in range(12)]
GROUP_MAP = {s: ("半導體" if i < 6 else "生技") for i, s in enumerate(STOCKS)}
FIELDS = ["f1", "f2", "f3"]


def make_data(seed=7):
    """合成月頻寬表面板：index=月份, columns=股票。含隨機缺值以測 NaN 路徑。"""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2015-01", periods=N_MONTHS, freq="M")
    data = {}
    for k, f in enumerate(FIELDS):
        arr = rng.normal(0, 1, size=(N_MONTHS, len(STOCKS))).cumsum(axis=0)
        mask = rng.random(arr.shape) < 0.05
        arr[mask] = np.nan
        data[f] = pd.DataFrame(arr, index=idx, columns=STOCKS)
    return data


@pytest.fixture
def data():
    return make_data()


@pytest.fixture
def group_map():
    return dict(GROUP_MAP)
