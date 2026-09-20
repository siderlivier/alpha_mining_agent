"""Versioned reference pool from all registered local base fields.
Selection, direction and dedup use sub_train/validation only; legacy DFS pools
are archived through Memory transactions. --apply replaces references atomically.
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


def select(cand, min_icir, max_decay, take_all=False):
    if take_all:
        raise ValueError("Unqualified reference admission is disabled")
    d = cand.copy()
    d["decay_pct"] = (1 - d.ICIR_validation.abs() / d.ICIR_train.abs().replace(0, np.nan)) * 100
    return d[(d.t_train.abs() > 2) & (d.ICIR_train.abs() > min_icir)
             & (np.sign(d.ICIR_train) == np.sign(d.ICIR_validation))
             & (d.ICIR_validation.abs() > 0.2) & (d.decay_pct <= max_decay)
             & (d.n_train >= 36) & (d.n_validation >= 12)]


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
    # Selection/dedup must never depend on sealed test values.
    lo, hi = CFG["split"]["sub_train"][0], CFG["split"]["validation"][1]
    long = long[long.ym.between(lo, hi)]
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


def do_clear(mem):
    with mem.locked():
        lib, values = mem.snapshot()
        keep = {k: v for k, v in lib.items() if not k.startswith(REFERENCE_PREFIX)}
        return mem.commit_snapshot(keep, values[values.factor_id.isin(keep)])


def reference_candidates(base):
    """Fixed catalog: every registered base field, no DFS survivor/test lists."""
    import eval_candidates as ec
    from factor_scope import context_for
    ctx = context_for(base, CFG)
    catalog = yaml.safe_load((ROOT / "fields.yaml").read_text(encoding="utf-8"))
    names = sorted(f for group in catalog.values() for f in group)
    rows, frames = [], []
    for name in names:
        fac = ctx.data[name]
        spans = {}
        for span in ("sub_train", "validation"):
            # Last label of each span would use a price outside that span.
            months = ctx.mask(span)
            hi = str(pd.Period(CFG["split"][span][1], freq="M") - 1)
            spans[span] = ec.monthly_ic(ctx, fac, [m for m in months if m <= hi]).dropna()
        tr, va = spans["sub_train"], spans["validation"]
        def score(x):
            return float(x.mean() / x.std()) if len(x)>1 and x.std()>0 else np.nan
        train, valid = score(tr), score(va)
        t = float(tr.mean()/tr.std()*np.sqrt(len(tr))) if len(tr)>1 and tr.std()>0 else np.nan
        rows.append(dict(factor=name, ICIR_train=train, ICIR_validation=valid,
                         t_train=t, n_train=len(tr), n_validation=len(va)))
        frame = base[["ym", "stock_id", name]].rename(columns={name: "value"}).dropna()
        frame.insert(0, "dfs_name", name)
        frames.append(frame)
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import hashlib
    ap = argparse.ArgumentParser(description="Versioned local-field reference pool; train/validation only")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--clear", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--list", action="store_true")
    ap.add_argument("--min-icir", type=float, default=0.3)
    ap.add_argument("--max-decay", type=float, default=50)
    ap.add_argument("--max-ref-corr", type=float, default=0.95)
    a = ap.parse_args()
    if not (0 <= a.min_icir and np.isfinite(a.min_icir) and 0 <= a.max_decay <= 100
            and 0 < a.max_ref_corr <= 1):
        raise ValueError("invalid reference thresholds")
    mem = Memory(ROOT / CFG["paths"]["memory_dir"])
    mem.ensure()
    if a.clear:
        print("Archived transaction:", do_clear(mem)); return
    # Serialize against mining/scope commits for the full replacement.
    with mem.locked():
        base_path = ROOT / CFG["paths"]["monthly_base"]
        raw = base_path.read_bytes()
        base_hash = hashlib.sha256(raw).hexdigest()
        base = pd.read_parquet(base_path)
        cand, long = reference_candidates(base)
        sel = select(cand, a.min_icir, a.max_decay).sort_values("ICIR_train", key=abs, ascending=False)
        names, dropped = dedupe_reference(long, sel.factor.tolist(), sel.set_index("factor"), a.max_ref_corr)
        print(f"Fixed base-field catalog: {len(cand)}; qualified: {len(sel)}; deduplicated: {len(names)}")
        print(sel[["factor", "ICIR_train", "ICIR_validation", "decay_pct"]].to_string(index=False))
        if not names:
            raise ValueError("Empty reference release; active pool unchanged")
        if not a.apply:
            print("Dry run; active pool unchanged"); return
        lib, old = mem.snapshot()
        prior_ids = [int(k[2:]) for k in lib if k.startswith(REFERENCE_PREFIX)]
        # Include archived IDs to prevent reuse even following --clear.
        for archived in (mem.root / "transactions").glob("*/library.before"):
            prior_ids += [int(k[2:]) for k in json.loads(archived.read_text(encoding="utf-8")) if k.startswith(REFERENCE_PREFIX)]
        next_id = max(prior_ids, default=0) + 1
        keep = {k: v for k, v in lib.items() if not k.startswith(REFERENCE_PREFIX)}
        values = [old[old.factor_id.isin(keep)]]
        row = sel.set_index("factor")
        release = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, name in enumerate(names, next_id):
            fid = f"R-{i:03d}"
            r = row.loc[name]
            orientation = 1 if r.ICIR_train > 0 else -1
            from memory import derive_aspect
            catalog = yaml.safe_load((ROOT / "fields.yaml").read_text(encoding="utf-8"))
            definition = next(group[name] for group in catalog.values() if name in group)
            desc = definition["desc"]
            zh, aspect = desc.split("（")[0], derive_aspect([name])
            if orientation < 0:
                zh, desc = "反向：" + zh, "原欄位取負後使用；" + desc
            keep[fid] = dict(name_zh=zh, desc_zh=desc, aspect=aspect, category="base_field_reference",
                reference=True, formula=None, fhash=None, fields=[name], depth=None,
                source="registered_base_fields_v1", source_field=name, reference_release=release,
                value_orientation=orientation, approved_groups=sorted(base.group.unique()),
                icir_train=float(r.ICIR_train)*orientation,
                icir_validation=float(r.ICIR_validation)*orientation, decay_pct=float(r.decay_pct),
                selection_periods={k: CFG["split"][k] for k in ("sub_train", "validation")},
                base_hash=base_hash, test_metrics_sealed=None,
                created_at=datetime.now().isoformat(timespec="seconds"))
            v = long[long.dfs_name.eq(name)].drop(columns="dfs_name").copy()
            v["value"] *= orientation
            v.insert(0,"factor_id",fid)
            values.append(v)
        if hashlib.sha256(base_path.read_bytes()).hexdigest() != base_hash:
            raise ValueError("Base changed during selection")
        transaction = mem.commit_snapshot(keep, pd.concat(values, ignore_index=True))
        print(f"Committed reference release {release}; {len(names)} factors; backup transaction {transaction}")


if __name__ == "__main__":
    main()
