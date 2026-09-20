"""
M1/v1.3：panel.parquet（日頻 PIT 面板）→ monthly_base.parquet（月頻基礎欄位快照）。

v1.3 起分三階段（記憶體與執行時間友善，各階段可獨立重跑）：
  --stage prices  價量技術面日頻衍生 → 月底快照 → data/tmp_prices.parquet
  --stage chips   籌碼面日頻衍生     → 月底快照 → data/tmp_chips.parquet
  --stage final   基本面快照 + 併三表 + 成長率/比率 + fwd_ret → monthly_base.parquet
  --stage all     依序全跑（預設，本機使用）

口徑說明見規格書 4.2 與附錄 C/D。所有滾動計算僅向後看。

執行：python src/build_base.py            # 全部
     python src/build_base.py --stage prices
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
PANEL = (ROOT / CFG["paths"]["panel"]).resolve()
OUT = ROOT / CFG["paths"]["monthly_base"]
TMP_P = ROOT / "data" / "tmp_prices.parquet"
TMP_C = ROOT / "data" / "tmp_chips.parquet"

KEYS = ["date", "stock_id"]


def load(cols):
    tbl = pq.read_table(PANEL, columns=cols)
    df = tbl.to_pandas()
    df["stock_id"] = df["stock_id"].astype("category")
    return df.sort_values(["stock_id", "date"]).reset_index(drop=True)


def snap_monthly(df):
    df = df.copy()
    df["ym"] = df["date"].dt.to_period("M").astype(str)
    m = df.groupby(["stock_id", "ym"], observed=True).tail(1)
    return m.sort_values(["stock_id", "ym"]).reset_index(drop=True)


def coalesce(df, *names):
    s = None
    for nm in names:
        if nm in df.columns:
            s = df[nm].copy() if s is None else s.fillna(df[nm])
    return s


def sdiv(a, b):
    return a / b.replace(0, np.nan)


def calendar_value(m, column, offset):
    """Same stock, exact calendar month. Missing months remain missing."""
    periods = pd.PeriodIndex(m["ym"], freq="M")
    keys = pd.MultiIndex.from_arrays([m["stock_id"].astype(str), periods])
    if keys.has_duplicates:
        raise ValueError("duplicate stock/month")
    values = pd.Series(m[column].to_numpy(dtype=float), index=keys)
    wanted = pd.MultiIndex.from_arrays([m["stock_id"].astype(str), periods + offset])
    return pd.Series(values.reindex(wanted).to_numpy(), index=m.index)


def issued_shares(df):
    """PIT exchange-issued shares, in shares; no assumed-par fallback."""
    shares = pd.to_numeric(df["shares_issued"], errors="coerce")
    return shares.where(np.isfinite(shares) & shares.gt(0))


def profit_growth(m, column, months=12):
    """同股票、精確前 N 個月的獲利變化／前值絕對值；零或缺值無定義。"""
    periods = pd.PeriodIndex(m["ym"], freq="M")
    keys = pd.MultiIndex.from_arrays([m["stock_id"].astype(str), periods])
    if keys.has_duplicates:
        raise ValueError("profit_growth 需要唯一的 stock_id/ym")
    values = pd.Series(m[column].to_numpy(dtype=float), index=keys)
    previous_keys = pd.MultiIndex.from_arrays(
        [m["stock_id"].astype(str), periods - months])
    previous = pd.Series(values.reindex(previous_keys).to_numpy(), index=m.index)
    return (m[column] - previous) / previous.abs().replace(0, np.nan) * 100


# ---------------------------------------------------------------------------

def stage_prices():
    df = load(KEYS + ["group", "close", "close_raw", "ret", "volume",
                      "amount", "b_CapitalStock", "shares_issued"])
    g = df.groupby("stock_id", observed=True, sort=False)
    for n in (20, 60, 120, 240):
        df[f"mom_{n}"] = g["close"].pct_change(n, fill_method=None)
    for n in (21, 63, 126):
        df[f"vol_{n}"] = g["ret"].transform(
            lambda s, n=n: s.rolling(n, min_periods=max(10, n // 2)).std())
    df["amt_21"] = g["amount"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    hi252 = g["close"].transform(lambda s: s.rolling(252, min_periods=120).max())
    df["px_hi252"] = df["close"] / hi252
    shares = issued_shares(df)
    df["turn_d"] = df["volume"] / shares.replace(0, np.nan)
    df["turn_21"] = g["turn_d"].transform(lambda s: s.rolling(21, min_periods=10).mean())

    keep = KEYS + ["group", "close", "close_raw", "b_CapitalStock", "shares_issued",
                   "mom_20", "mom_60", "mom_120", "mom_240",
                   "vol_21", "vol_63", "vol_126", "amt_21", "px_hi252", "turn_21"]
    snap_monthly(df[keep]).to_parquet(TMP_P, index=False)
    print(f"✅ prices → {TMP_P}")


def stage_chips():
    df = load(KEYS + ["inst_foreign_net", "inst_trust_net", "margin_balance",
                      "short_balance", "foreign_ratio", "lending_vol",
                      "div_yield", "volume", "b_CapitalStock", "shares_issued"])
    g = df.groupby("stock_id", observed=True, sort=False)
    shares = issued_shares(df)
    df["_fn"] = df["inst_foreign_net"] / shares
    df["frgn_net_21"] = g["_fn"].transform(lambda s: s.rolling(21, min_periods=10).sum())
    df["_tn"] = df["inst_trust_net"] / shares
    df["trust_net_21"] = g["_tn"].transform(lambda s: s.rolling(21, min_periods=10).sum())
    df["frgn_ratio"] = df["foreign_ratio"]
    df["frgn_ratio_d63"] = g["foreign_ratio"].transform(lambda s: s.diff(63))
    df["margin_d21"] = g["margin_balance"].transform(
        lambda s: s.pct_change(21, fill_method=None))
    df["short_margin_ratio"] = df["short_balance"] / df["margin_balance"].replace(0, np.nan)
    lend_ma = g["lending_vol"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    vol_ma = g["volume"].transform(lambda s: s.rolling(21, min_periods=10).mean())
    df["lend_vol_21"] = lend_ma / vol_ma.replace(0, np.nan)

    keep = KEYS + ["frgn_net_21", "trust_net_21", "frgn_ratio", "frgn_ratio_d63",
                   "margin_d21", "short_margin_ratio", "lend_vol_21", "div_yield"]
    snap_monthly(df[keep]).to_parquet(TMP_C, index=False)
    print(f"✅ chips → {TMP_C}")


def stage_final():
    df = load(KEYS + ["month_revenue",
                      "f_Revenue", "f_GrossProfit", "f_OperatingIncome", "f_EPS",
                      "f_NetIncome", "f_IncomeAfterTaxes", "f_IncomeAfterTax",
                      "f_TotalConsolidatedProfitForThePeriodAfterTax",
                      "b_Equity", "b_EquityAttributableToOwnersOfParent",
                      "b_TotalAssets",
                      "c_CashFlowsFromOperatingActivities"])
    m = snap_monthly(df)
    del df
    p = pd.read_parquet(TMP_P)
    c = pd.read_parquet(TMP_C)
    m = (m.drop(columns=["date"])
          .merge(p.drop(columns=["date"]), on=["stock_id", "ym"], how="right")
          .merge(c.drop(columns=["date"]), on=["stock_id", "ym"], how="left"))
    m = m.sort_values(["stock_id", "ym"]).reset_index(drop=True)
    gm = m.groupby("stock_id", observed=True, sort=False)

    ni = coalesce(m, "f_NetIncome", "f_IncomeAfterTaxes", "f_IncomeAfterTax",
                  "f_TotalConsolidatedProfitForThePeriodAfterTax")
    rev_q = m["f_Revenue"]
    px_raw = m["close_raw"].where(m["close_raw"].gt(0))
    mktcap = px_raw * issued_shares(m)

    m["roe"] = sdiv(ni, m["b_Equity"])
    m["gross_margin"] = sdiv(m["f_GrossProfit"], rev_q)
    m["op_margin"] = sdiv(m["f_OperatingIncome"], rev_q)
    m["net_margin"] = sdiv(ni, rev_q)
    m["accruals"] = sdiv(ni - m["c_CashFlowsFromOperatingActivities"], m["b_TotalAssets"])
    m["ocf_ratio"] = sdiv(m["c_CashFlowsFromOperatingActivities"], ni)
    m["ep"] = sdiv(m["f_EPS"], px_raw)
    m["bp"] = sdiv(m["b_EquityAttributableToOwnersOfParent"], mktcap)
    m["sp"] = sdiv(rev_q, mktcap)

    m["rev_yoy"] = (sdiv(m["month_revenue"], calendar_value(m, "month_revenue", -12)) - 1) * 100
    gm = m.groupby("stock_id", observed=True, sort=False)
    m["rev_yoy_ma3"] = gm["rev_yoy"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    m["rev_mom"] = (sdiv(m["month_revenue"], calendar_value(m, "month_revenue", -1)) - 1) * 100
    m["eps_yoy"] = profit_growth(m, "f_EPS")
    m["op_income_yoy"] = profit_growth(m, "f_OperatingIncome")

    # 目標變數：未來一月報酬（還原價），逐月 winsorize 壓假極端
    m["fwd_ret_1m"] = sdiv(calendar_value(m, "close", 1), m["close"]) - 1
    m["fwd_ret_1m"] = m.groupby("ym")["fwd_ret_1m"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))

    field_yaml = yaml.safe_load((ROOT / "fields.yaml").read_text(encoding="utf-8"))
    fields = [f for grp in field_yaml.values() for f in grp]
    keep = ["stock_id", "ym", "group", "fwd_ret_1m"] + fields
    missing = [x for x in keep if x not in m.columns]
    if missing:
        raise SystemExit(f"❌ 缺少欄位: {missing}")
    out = m[keep].copy()
    out["stock_id"] = out["stock_id"].astype(str)
    out["group"] = out["group"].astype(str)
    num_cols = out.columns.difference(["stock_id", "ym", "group"])
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)

    print("\n欄位缺值率：")
    for f in fields:
        na = out[f].isna().mean()
        print(f"  {f:20s} {na:6.1%}{'  ⚠️' if na > 0.4 else ''}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n✅ 輸出 {OUT}  shape={out.shape}, "
          f"月份 {out['ym'].min()} ~ {out['ym'].max()}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "prices", "chips", "final"])
    a = ap.parse_args()
    if a.stage in ("all", "prices"):
        stage_prices()
    if a.stage in ("all", "chips"):
        stage_chips()
    if a.stage in ("all", "final"):
        stage_final()


if __name__ == "__main__":
    main()
