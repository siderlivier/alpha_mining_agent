"""
把前置專案（tw_alpha_strategy）DFS 挖出的因子，匯入本專案的因子庫作為
「參考因子」（R-xxx），讓 Stage 2 也對它們去相關。

為什麼需要
----------
agent 的 31 個欄位裡有 24 個直接來自或等價於 mine_dfs.py 的因子，但 Stage 2
原本只跟 agent「自己入庫的」因子去相關。結果是：一個候選通過 Stage 2，只證明
它跟 agent 已有的因子不重複，**不保證跟你早就知道的 86 個 DFS 因子不重複**。
實際發生過：F-005 `cs_rank(neg(vol_126))` 就是 `neg_vol_126`，F-006 幾乎等於
`neg_accruals`。

存放位置（單一因子庫）
----------------------
  memory/library.json           R-xxx 與 F-xxx 同住，靠 reference:true 區分
  memory/factor_values.parquet  月頻值（Stage 2 的資料源）
  data/dfs_snapshot.parquet     自足快照 ← 有它之後就不必再跨專案

跨專案只發生在「第一次匯入」。之後 --clear / 重新匯入都直接讀快照，
不需要 tw_alpha_strategy 的程式或 panel.parquet。

篩選（三道）
------------
  1. t_train > 2 且測試期同向、|ICIR_test| > 0.2   ← 沿用前置專案自己的判準
  2. |ICIR_train| > --min-icir（預設 0.3）
  3. 衰減 ≤ --max-decay（預設 50%）                ← 與本專案 Stage 4 同一標準
訓練期漂亮、測試期崩掉的因子不該當作「已知因子」去擋別人。

洩漏紀律
--------
  - 參考因子在 Generate prompt 中**只給中文名與一句話描述，不給公式**：
    它們是用原始財報欄位算的（gp_to_px = 毛利/股價），本 DSL 沒有那些欄位，
    給了只會誘使模型拼出無效公式。列出的目的是「別再提相近變體」。
  - report / audit / 統計一律走 Memory.own_library()，不把參考因子算成成果。

用法
----
    python src/seed_reference.py --list       # 只看篩選結果，不計算
    python src/seed_reference.py --dry-run    # 計算但不寫（第一次會讀前置專案）
    python src/seed_reference.py --apply
    python src/seed_reference.py --apply --max-decay 40 --min-icir 0.4
    python src/seed_reference.py --clear      # 移除所有 R-xxx
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory import REFERENCE_PREFIX, Memory

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SNAPSHOT = ROOT / "data" / "dfs_snapshot.parquet"

# DFS 因子 → 中文名與白話描述。
# 沒列到的會退回 "[參考] <原名>"，不會漏掉，只是描述較粗。
NAMES = {
    "gp_to_px": ("毛利股價比", "毛利 / 股價，價值面的獲利含量"),
    "op_to_px": ("營益股價比", "營業利益 / 股價"),
    "ni_to_px": ("淨利股價比", "稅後淨利 / 股價"),
    "pretax_to_px": ("稅前股價比", "稅前淨利 / 股價"),
    "eps_to_px": ("每股盈餘股價比", "EPS / 股價，盈餘殖利率"),
    "cfo_yield": ("營運現金流殖利率", "營運現金流 / 市值"),
    "fcf_yield": ("自由現金流殖利率", "(營運現金流 − 資本支出) / 市值"),
    "bp": ("淨值股價比", "股東權益 / 市值"),
    "div_yield": ("現金殖利率", "現金股利 / 股價"),
    "ep": ("盈餘殖利率", "1 / 本益比"),
    "gp_to_rev": ("毛利率", "毛利 / 營收"),
    "op_to_rev": ("營益率", "營業利益 / 營收"),
    "ni_to_rev": ("淨利率", "稅後淨利 / 營收"),
    "pretax_to_rev": ("稅前淨利率", "稅前淨利 / 營收"),
    "gross_margin": ("毛利率", "毛利 / 營收"),
    "op_margin": ("營益率", "營業利益 / 營收"),
    "net_margin": ("淨利率", "稅後淨利 / 營收"),
    "d_gross_margin": ("毛利率年變化", "毛利率的 12 個月變化"),
    "d_op_margin": ("營益率年變化", "營益率的 12 個月變化"),
    "d_net_margin": ("淨利率年變化", "淨利率的 12 個月變化"),
    "roe": ("股東權益報酬率", "稅後淨利 / 股東權益"),
    "roa": ("資產報酬率", "稅後淨利 / 總資產"),
    "d_roe": ("ROE 年變化", "ROE 的 12 個月變化"),
    "d_roa": ("ROA 年變化", "ROA 的 12 個月變化"),
    "asset_turnover": ("資產週轉率", "營收 / 總資產"),
    "d_asset_turnover": ("資產週轉率年變化", "資產週轉率的 12 個月變化"),
    "neg_accruals": ("低應計品質", "−(淨利 − 營運現金流) / 總資產，低應計代表高品質"),
    "neg_debt_ratio": ("低負債比", "−負債 / 總資產"),
    "d_debt_ratio": ("負債比年變化", "負債比的 12 個月變化"),
    "current_ratio": ("流動比率", "流動資產 / 流動負債"),
    "neg_asset_growth": ("低資產成長", "−總資產的 12 個月成長率，資產擴張過快的懲罰"),
    "quality_score": ("Piotroski 品質分", "ROA/現金流/應計/趨勢等訊號的加總"),
    "cfo_to_ni": ("現金流品質", "營運現金流 / 淨利"),
    "capex_to_ta": ("資本支出強度", "資本支出 / 總資產"),
    "capex_to_rev": ("資本支出營收比", "資本支出 / 營收"),
    "g_cfo_12": ("營運現金流年增", "營運現金流的 12 個月成長率"),
    "inv_turnover": ("存貨週轉率", "營收 / 存貨"),
    "recv_turnover": ("應收週轉率", "營收 / 應收帳款"),
    "d_inv_turnover": ("存貨週轉年變化", "存貨週轉率的 12 個月變化"),
    "d_recv_turnover": ("應收週轉年變化", "應收週轉率的 12 個月變化"),
    "cash_to_ta": ("現金資產比", "現金及約當現金 / 總資產"),
    "inv_to_ta": ("存貨資產比", "存貨 / 總資產"),
    "recv_to_ta": ("應收資產比", "應收帳款 / 總資產"),
    "ppe_to_ta": ("固定資產比", "不動產廠房設備 / 總資產"),
    "intang_to_ta": ("無形資產比", "無形資產 / 總資產"),
    "retain_to_ta": ("保留盈餘比", "保留盈餘 / 總資產"),
    "curr_to_ta": ("流動資產比", "流動資產 / 總資產"),
    "d_nwc_to_ta": ("淨營運資金變化", "(流動資產−流動負債)/總資產 的 12 個月變化"),
    "mom_1m": ("一月動能", "近 1 個月報酬"),
    "mom_3m": ("三月動能", "近 3 個月報酬"),
    "mom_6m": ("半年動能", "近 6 個月報酬"),
    "mom_12m": ("一年動能", "近 12 個月報酬"),
    "mom_12_1": ("動能（跳過最近一月）", "12 個月報酬但跳過最近 1 月，避開短期反轉"),
    "mom_6_1": ("半年動能（跳過最近一月）", "6 個月報酬跳過最近 1 月"),
    "mom_9_1": ("九月動能（跳過最近一月）", "9 個月報酬跳過最近 1 月"),
    "mom_vol_adj": ("風險調整動能", "動能 / 波動度"),
    "dist_high": ("距年高點", "股價 / 252 日高點 − 1，越接近新高越大"),
    "liq_amt": ("流動性", "近 21 日成交金額的對數"),
    "neg_vol_21": ("低波動（月）", "−21 日報酬標準差"),
    "neg_vol_63": ("低波動（季）", "−63 日報酬標準差"),
    "neg_vol_126": ("低波動（半年）", "−126 日報酬標準差"),
    "neg_size": ("小型股溢酬", "−市值取對數"),
    "neg_short_ratio": ("低券資比", "−借券餘額 / 融資餘額"),
    "rev_accel": ("營收成長加速", "3 個月營收成長率的變化"),
}
GROW_PREFIX = {"g_mrev": "月營收", "g_rev": "營收", "g_ni": "淨利",
               "g_op": "營業利益", "g_gp": "毛利", "g_eps": "EPS"}
ASPECT = {"mom": "技術面", "vol": "技術面", "dist": "技術面", "liq": "技術面",
          "size": "技術面", "short": "籌碼面"}


def describe(name: str) -> tuple[str, str, str]:
    """回傳 (中文名, 描述, 面向)。"""
    if name in NAMES:
        zh, desc = NAMES[name]
    else:
        base = name.rsplit("_", 1)
        if len(base) == 2 and base[0] in GROW_PREFIX and base[1].isdigit():
            zh = f"{GROW_PREFIX[base[0]]}{base[1]}期成長"
            desc = f"{GROW_PREFIX[base[0]]}的 {base[1]} 個月成長率"
        else:
            zh, desc = f"[參考] {name}", f"前置專案 DFS 因子 {name}"
    aspect = "基本面"
    for k, v in ASPECT.items():
        if k in name:
            aspect = v
            break
    return zh, desc, aspect


# ---------------------------------------------------------------------------

def upstream_root() -> Path:
    return (ROOT / CFG["paths"]["panel"]).resolve().parents[2]


def select(cand: pd.DataFrame, min_icir: float, max_decay: float,
           take_all: bool) -> pd.DataFrame:
    d = cand.copy()
    d["decay_pct"] = (1 - d["ICIR_test"].abs()
                      / d["ICIR_train"].abs().replace(0, np.nan)) * 100
    if take_all:
        return d
    ok = ((d["t_train"].abs() > 2)
          & (np.sign(d["ICIR_train"]) == np.sign(d["ICIR_test"]))
          & (d["ICIR_test"].abs() > 0.2)
          & (d["ICIR_train"].abs() > min_icir)
          & (d["decay_pct"] <= max_decay))
    return d[ok]


def compute_from_upstream(names: list[str]) -> pd.DataFrame:
    """跑一次前置專案的 generate()，回傳對齊後的長格式值，並存成自足快照。"""
    up = upstream_root()
    src = up / "src"
    if not (src / "mine_dfs.py").exists():
        raise SystemExit(
            f"找不到 {src / 'mine_dfs.py'}，也沒有 {SNAPSHOT}。\n"
            f"第一次匯入需要前置專案；之後只讀快照。")
    sys.path.insert(0, str(src))
    import mine_dfs

    panel_p = up / "data" / "processed" / "panel.parquet"
    print(f"讀取 {panel_p}（第一次匯入才需要，之後走快照）…")
    panel = pd.read_parquet(panel_p)
    print("建立月底基礎欄位…")
    m = mine_dfs.build_monthly_base(panel)
    print("生成 DFS 因子（沿用前置專案的 generate()，不重寫以免語意漂移）…")
    df, _ = mine_dfs.generate(m)

    mb = pd.read_parquet(ROOT / CFG["paths"]["monthly_base"])
    months = set(mb["ym"].astype(str))
    stocks = set(mb["stock_id"].astype(str))
    df = df.copy()
    df["ym"] = df["ym"].astype(str)
    df["stock_id"] = df["stock_id"].astype(str)
    df = df[df["ym"].isin(months) & df["stock_id"].isin(stocks)]

    have = [n for n in names if n in df.columns]
    miss = [n for n in names if n not in df.columns]
    if miss:
        print(f"⚠️ 生成結果中沒有這些欄位，略過：{miss}")

    out = []
    for n in have:
        s = df[["ym", "stock_id", n]].rename(columns={n: "value"})
        s = s[s["value"].notna()]
        s.insert(0, "dfs_name", n)
        out.append(s)
    long = pd.concat(out, ignore_index=True)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(SNAPSHOT, index=False)
    print(f"✅ 自足快照已存：{SNAPSHOT.relative_to(ROOT)}"
          f"（{long['dfs_name'].nunique()} 個因子、{len(long):,} 筆）")
    print("   之後 --clear 或重新匯入都直接讀它，不再需要前置專案。")
    return long


def load_values(names: list[str], force_upstream: bool) -> pd.DataFrame:
    if SNAPSHOT.exists() and not force_upstream:
        long = pd.read_parquet(SNAPSHOT)
        have = set(long["dfs_name"])
        miss = [n for n in names if n not in have]
        if not miss:
            print(f"讀取自足快照 {SNAPSHOT.relative_to(ROOT)}（不需要前置專案）")
            return long[long["dfs_name"].isin(names)]
        print(f"快照缺少 {len(miss)} 個因子（{miss[:5]}…），改從前置專案重算。")
    return compute_from_upstream(names)


def dedupe_reference(long: pd.DataFrame, names: list[str], row: pd.DataFrame,
                     max_corr: float) -> tuple[list[str], list[tuple]]:
    """
    參考因子彼此之間去重複。

    為什麼需要：前置專案的 `mine_dfs.py` 有同一個公式掛兩個名字的情況
    （`net_margin` / `ni_to_rev` 都是稅後淨利 ÷ 營收，ρ = 1.000；
    `op_margin` / `op_to_rev` 同理）。原本的三道篩選是**逐因子**的，
    看不到這種兩兩重複——結果是：

      - 因子庫的參考因子數被灌水
      - factor_lab 等權合成時，淨利率這個概念被賦予兩倍權重

    做法與 Stage 2 同口徑：逐月橫斷面 Spearman，跨月取平均。
    按 |ICIR_train| 由大到小逐一檢視，與已保留者相關度超過門檻就丟掉。

    回傳 (保留的名稱, [(丟掉的, 因為誰, ρ), …])
    """
    wide = long.pivot_table(index=["ym", "stock_id"], columns="dfs_name",
                            values="value", aggfunc="first")
    wide = wide.reindex(columns=[n for n in names if n in wide.columns])
    ranks = wide.groupby(level="ym").rank()
    corr = ranks.corr().abs()          # 月內排名 → 池化相關，近似逐月平均

    order = sorted(wide.columns,
                   key=lambda n: abs(row.loc[n, "ICIR_train"]), reverse=True)
    keep, dropped = [], []
    for n in order:
        hit = next(((k, float(corr.loc[n, k])) for k in keep
                    if corr.loc[n, k] > max_corr), None)
        if hit:
            dropped.append((n, hit[0], hit[1]))
        else:
            keep.append(n)
    return [n for n in names if n in keep], dropped


def do_clear(mem: Memory):
    lib = mem.library()
    refs = [k for k in lib if k.startswith(REFERENCE_PREFIX)]
    for k in refs:
        lib.pop(k)
    mem._save_library(lib)
    if mem.values_path.exists():
        fv = pd.read_parquet(mem.values_path)
        n0 = fv["factor_id"].nunique()
        fv = fv[~fv["factor_id"].astype(str).str.startswith(REFERENCE_PREFIX)]
        fv.to_parquet(mem.values_path, index=False)
        print(f"factor_values.parquet：{n0} → {fv['factor_id'].nunique()} 個因子")
    print(f"已從 library.json 移除 {len(refs)} 個參考因子。"
          f"（快照 {SNAPSHOT.name} 保留，隨時可重新匯入）")


def main():
    ap = argparse.ArgumentParser(description="匯入前置專案 DFS 因子作為參考因子")
    ap.add_argument("--list", action="store_true", help="只列出篩選結果")
    ap.add_argument("--dry-run", action="store_true", help="計算但不寫檔")
    ap.add_argument("--apply", action="store_true", help="寫入")
    ap.add_argument("--clear", action="store_true", help="移除所有 R-xxx")
    ap.add_argument("--all", action="store_true", help="不篩選，全部匯入")
    ap.add_argument("--min-icir", type=float, default=0.3)
    ap.add_argument("--max-decay", type=float, default=50.0,
                    help="train→test ICIR 衰減上限%%（預設 50，與 Stage 4 同標準）")
    ap.add_argument("--max-ref-corr", type=float, default=0.95,
                    help="參考因子彼此的相關度上限（預設 0.95，只擋近乎重複的；"
                         "設 1.0 = 不去重）")
    ap.add_argument("--from-upstream", action="store_true",
                    help="忽略快照，強制重新從前置專案計算")
    a = ap.parse_args()

    mem = Memory()
    mem.ensure()
    if a.clear:
        do_clear(mem)
        return

    up = upstream_root()
    cand_p = up / "data" / "processed" / "dfs_candidates.csv"
    local_cand = ROOT / "data" / "dfs_candidates.csv"
    if local_cand.exists() and not a.from_upstream:
        cand = pd.read_csv(local_cand)
    elif cand_p.exists():
        cand = pd.read_csv(cand_p)
        local_cand.parent.mkdir(parents=True, exist_ok=True)
        cand.to_csv(local_cand, index=False)     # 複製一份，之後不必跨專案
        print(f"（已複製 dfs_candidates.csv 到 {local_cand.relative_to(ROOT)}）")
    else:
        raise SystemExit(f"找不到 dfs_candidates.csv（{cand_p} 或 {local_cand}）")

    sel = select(cand, a.min_icir, a.max_decay, a.all).sort_values(
        "ICIR_train", key=abs, ascending=False)
    dropped = len(cand) - len(sel)
    print(f"DFS 候選 {len(cand)} 個 → 通過篩選 {len(sel)} 個（淘汰 {dropped}）")
    if not a.all:
        print(f"  篩選：t_train>2、測試同向、|ICIR_test|>0.2、"
              f"|ICIR_train|>{a.min_icir}、衰減≤{a.max_decay:.0f}%\n")
    fmt = {"ICIR_train": "{:+.3f}".format, "ICIR_test": "{:+.3f}".format,
           "decay_pct": "{:.0f}%".format}
    print(sel[["factor", "ICIR_train", "ICIR_test", "decay_pct"]].head(40)
          .to_string(index=False, formatters=fmt))
    if len(sel) > 40:
        print(f"…另有 {len(sel) - 40} 個")

    if not a.all:
        cut = select(cand, a.min_icir, 999, False)
        bad = cut[~cut["factor"].isin(sel["factor"])]
        if len(bad):
            print(f"\n因衰減 > {a.max_decay:.0f}% 被淘汰的 {len(bad)} 個：")
            print(bad[["factor", "ICIR_train", "ICIR_test", "decay_pct"]]
                  .to_string(index=False, formatters=fmt))

    own_n = len(mem.own_library())
    print(f"\nStage 2：目前 {own_n} 個自有因子 → 匯入後共 {own_n + len(sel)} 個")

    if a.list:
        print("\n--list 模式，未計算因子值。")
        return

    names = sel["factor"].tolist()
    long = load_values(names, a.from_upstream)
    have = [n for n in names if n in set(long["dfs_name"])]
    row = sel.set_index("factor")

    # 第四道：參考因子彼此去重複（前置專案有同公式雙名的情況）
    if a.max_ref_corr < 1.0 and len(have) > 1:
        have, dup = dedupe_reference(long, have, row, a.max_ref_corr)
        if dup:
            print(f"\n參考因子互相去重（|ρ| > {a.max_ref_corr:.2f}）："
                  f"丟掉 {len(dup)} 個")
            for n, k, r in dup:
                print(f"   {n:18} ≈ {k:18} ρ={r:.3f}")
            long = long[long["dfs_name"].isin(have)]

    ids = {n: f"{REFERENCE_PREFIX}{i:03d}" for i, n in enumerate(have, 1)}

    if not a.apply:
        print(f"\n--dry-run：可匯入 {len(have)} 個因子、{len(long):,} 筆值，未寫入。")
        for n in have[:5]:
            zh, desc, asp = describe(n)
            print(f"   {ids[n]} {n:18} → 【{zh}】({asp}) {desc}")
        return

    lib = {k: v for k, v in mem.library().items()
           if not k.startswith(REFERENCE_PREFIX)}
    for n in have:
        zh, desc, asp = describe(n)
        r = row.loc[n]
        lib[ids[n]] = {
            "name_zh": zh, "desc_zh": desc, "aspect": asp,
            "category": "dfs_reference",
            "formula": None,          # ⚠️ 非 DSL 表達式，prompt 也刻意不給
            "fhash": None, "fields": [], "depth": None,
            "reference": True, "dfs_name": n,
            "source": "tw_alpha_strategy/src/mine_dfs.py",
            "industry_scope": None, "industry_metrics": None,
            "icir_train": float(r["ICIR_train"]), "icir_test": float(r["ICIR_test"]),
            "decay_pct": float(r["decay_pct"]),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "test_metrics_sealed": None,
        }
    mem._save_library(lib)

    vals = long[long["dfs_name"].isin(have)].copy()
    vals["factor_id"] = vals["dfs_name"].map(ids)
    vals = vals[["factor_id", "ym", "stock_id", "value"]]
    old = (pd.read_parquet(mem.values_path) if mem.values_path.exists()
           else pd.DataFrame(columns=vals.columns))
    old = old[~old["factor_id"].astype(str).str.startswith(REFERENCE_PREFIX)]
    merged = pd.concat([old, vals], ignore_index=True)
    mem.values_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(mem.values_path, index=False)

    print(f"\n✅ library.json：{len(mem.own_library())} 個自有 + "
          f"{len(mem.reference_library())} 個參考因子")
    print(f"✅ factor_values.parquet：{merged['factor_id'].nunique()} 個因子、"
          f"{len(merged):,} 筆值")
    print("\n下一輪 Generate 會在因子庫摘要看到「已知因子」清單（只有名稱與描述、"
          "不給公式）；Stage 2 會對它們去相關。")
    print("要還原：python src/seed_reference.py --clear")


if __name__ == "__main__":
    main()
