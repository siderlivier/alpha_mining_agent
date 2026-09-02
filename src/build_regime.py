"""
M1：建立市場狀態標註表 regime_table.json + 月頻大盤報酬 market_monthly.parquet。

規格書 4.4：逐年 TAIEX 報酬/波動由 benchmarks.parquet 計算（客觀），
多空標籤依報酬門檻自動生成（>+10% 多頭、<-10% 空頭、其餘盤整），
風格備註為人工知識（一次性維護，evaluator 輸出逐年 IC 時 join 此表）。

執行：python src/build_regime.py
     python src/build_regime.py --dry-run   # 只印結果，不寫檔
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

BENCH = (ROOT / CFG["paths"]["panel"]).resolve().parents[1] / "raw" / "benchmarks.parquet"

# 人工風格備註（台股市場常識，與資料計算的報酬互補）
STYLE_NOTES = {
    2012: "歐債危機後回穩，證所稅爭議壓抑量能",
    2013: "QE 資金行情，電子復甦",
    2014: "溫和多頭，蘋概股主導",
    2015: "8 月人民幣貶值股災急跌，全年偏空",
    2016: "電子復甦，外資回流",
    2017: "大型權值股（台積電/蘋概）獨強，中小型落後",
    2018: "中美貿易戰，Q4 急跌",
    2019: "貿易戰轉單效應、台商回流，全年大多頭",
    2020: "COVID 3 月崩跌後 V 型反轉，電子/航運起漲",
    2021: "航運/半導體大行情，散戶參與度暴增",
    2022: "聯準會暴力升息，全年空頭，成長股重挫",
    2023: "AI 熱潮啟動，台積電/AI 供應鏈領漲",
    2024: "AI 資本支出潮，大型股獨強、集中度極高",
    2025: "（人工待補：請依實際行情補充風格備註）",
    2026: "（人工待補：請依實際行情補充風格備註）",
}


def label(ret):
    if np.isnan(ret):
        return "資料不足"
    if ret > 0.10:
        return "多頭"
    if ret < -0.10:
        return "空頭"
    return "盤整"


def main():
    ap = argparse.ArgumentParser(description="建立市場狀態表與月頻大盤報酬")
    ap.add_argument("--dry-run", action="store_true", help="只印結果，不寫檔")
    a = ap.parse_args()

    if not BENCH.exists():
        raise SystemExit(f"找不到基準資料 {BENCH}\n請先在前置專案執行 fetch_benchmarks.py")
    b = pd.read_parquet(BENCH)
    b["year"] = b["ym"].str[:4].astype(int)

    # 月頻大盤報酬 → 供 evaluator 計算條件 IC（上漲月 vs 下跌月）
    mkt = b[["ym", "BM_TAIEX_TR"]].rename(columns={"BM_TAIEX_TR": "taiex_ret"})
    out_mkt = ROOT / "data" / "market_monthly.parquet"
    if not a.dry_run:
        out_mkt.parent.mkdir(parents=True, exist_ok=True)
        mkt.to_parquet(out_mkt, index=False)

    table = {}
    for y, gdf in b.groupby("year"):
        r = gdf["BM_TAIEX_TR"].dropna()
        ann_ret = float((1 + r).prod() - 1) if len(r) else float("nan")
        ann_vol = float(r.std() * np.sqrt(12)) if len(r) > 3 else float("nan")
        table[str(y)] = {
            "taiex_ret": round(ann_ret, 4) if not np.isnan(ann_ret) else None,
            "taiex_vol": round(ann_vol, 4) if not np.isnan(ann_vol) else None,
            "regime": label(ann_ret),
            "note": STYLE_NOTES.get(y, ""),
        }

    out = ROOT / CFG["paths"]["regime_table"]
    if a.dry_run:
        print("（--dry-run：未寫入任何檔案）")
    else:
        out.write_text(json.dumps(table, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"✅ {out}")
    for y, v in table.items():
        print(f"  {y}: {v['regime']:4s} ret={v['taiex_ret']} | {v['note'][:24]}")


if __name__ == "__main__":
    main()
