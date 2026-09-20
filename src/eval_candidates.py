"""
M1：四階段漏斗評估器（規格書第 6 章）。

輸入：候選批次 JSON
  [{"id","formula","category","hypothesis","prediction","direction":"pos|neg"}, ...]
輸出：每個候選一份診斷 JSON（緊湊，~300 token，規格 6.4 schema）

階段：
  Stage 0 語法/複雜度（dsl.parse）→ rejected_syntax
  Stage 1 sub-train 快速 IC + 假設方向一致性 → rejected_stage1
  Stage 2 與因子庫去相關（|ρ| > 0.5，點名兇手）→ rejected_stage2
  Stage 3 批內去重（|ρ| > 0.7 留 ICIR 高者）→ rejected_stage3
  Stage 4 完整驗證（ICIR/衰減/多頭腿/覆蓋率/換手率）→ rejected_stage4 / passed

⛔ 洩漏紀律（規格 9.1）：本模組一切統計只用 sub-train + validation。
   test 期指標僅在 --seal 模式下計算，寫入密封欄位，絕不進 stdout/診斷輸出。

執行：
  python src/eval_candidates.py --in batch.json --out diags.json
  python src/eval_candidates.py --self-test          # 內建煙霧測試
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsl
from memory import Memory, approved_groups

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
FUN = CFG["funnel"]
MIN_STOCKS = FUN["min_stocks_per_group"]


def _r(x, nd=3):
    """診斷輸出的數字瘦身（token 密度）。"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def orient_factor(fac, metadata):
    """Stored values use this orientation; legacy metadata means raw (+1)."""
    orientation = metadata.get("value_orientation", 1)
    if orientation not in (-1, 1):
        raise ValueError("value_orientation must be +1 or -1")
    return fac * orientation


def trading_use(legs):
    """用途取決於定向後兩腿的超額，不以 direction 直接推定可做空。"""
    long_ok = (legs.get("long_excess_ann") or 0) > 0
    short_ok = (legs.get("short_excess_ann") or 0) > 0
    return ("long_short" if long_ok and short_ok else
            "long_only" if long_ok else "short_only" if short_ok else "unusable")


# ---------------------------------------------------------------------------
# 資料上下文（載入一次，整批共用）
# ---------------------------------------------------------------------------

class Context:
    def __init__(self, base=None, load_library=True, memory_root=None):
        self.memory_root = Path(memory_root) if memory_root else ROOT / CFG["paths"]["memory_dir"]
        mb = pd.read_parquet(ROOT / CFG["paths"]["monthly_base"]) if base is None else base
        fields = [c for c in mb.columns
                  if c not in ("stock_id", "ym", "group", "fwd_ret_1m")]
        self.months = sorted(mb["ym"].unique())
        piv = lambda c: (mb.pivot_table(index="ym", columns="stock_id",
                                        values=c, aggfunc="first")
                         .reindex(self.months))
        self.data = {f: piv(f) for f in fields}
        self.fwd = piv("fwd_ret_1m")
        # 對齊所有寬表的 columns（Engine 要求一致）
        cols = pd.Index(sorted(mb["stock_id"].unique()))
        self.fwd = self.fwd.reindex(columns=cols)
        self.data = {k: v.reindex(columns=cols) for k, v in self.data.items()}
        self.group_map = mb.groupby("stock_id")["group"].last().to_dict()
        self.fields = set(fields)

        s = CFG["split"]
        self.win = {
            "sub_train": (s["sub_train"][0], s["sub_train"][1]),
            "validation": (s["validation"][0], s["validation"][1]),
            "test": (s["test"][0], s["test"][1]),
        }
        self.regime = json.loads(
            (ROOT / CFG["paths"]["regime_table"]).read_text(encoding="utf-8"))
        mkt = pd.read_parquet(ROOT / "data" / "market_monthly.parquet")
        self.mkt = mkt.set_index("ym")["taiex_ret"]

        self._cols = cols
        self.lib_values, self.lib_meta = None, {}
        if load_library:
            self.reload_library()

    def reload_library(self):
        """載入/刷新因子庫（入庫後呼叫，讓 Stage 2 立即看到新因子）。"""
        self.lib_values, self.lib_meta = None, {}
        fv = self.memory_root / "factor_values.parquet"
        lj = self.memory_root / "library.json"
        if fv.exists():
            self.lib_meta, lv = Memory(fv.parent).snapshot()
            if len(lv):
                self.lib_values = {
                    fid: g.pivot_table(index="ym", columns="stock_id",
                                       values="value", aggfunc="first")
                         .reindex(self.months).reindex(columns=self._cols)
                    for fid, g in lv.groupby("factor_id")}
                from memory import approved_groups
                for fid, values in self.lib_values.items():
                    scope = approved_groups(self.lib_meta.get(fid, {}))
                    if scope is not None:
                        outside = [s for s in values.columns if self.group_map.get(s) not in scope]
                        values.loc[:, outside] = np.nan
        if lj.exists() and not self.lib_meta:
            # library.json 同時含 agent 自有的 F-xxx 與匯入的參考因子 R-xxx
            # （見 src/seed_reference.py）。兩者都要參與 Stage 2 去相關，
            # 所以這裡全載入；「哪些算 agent 的成果」由 Memory.own_library() 區分。
            self.lib_meta = Memory(lj.parent).library()

    def mask(self, span):
        lo, hi = self.win[span]
        return [m for m in self.months if lo <= m <= hi]


# ---------------------------------------------------------------------------
# 統計核心
# ---------------------------------------------------------------------------

def group_cols(ctx, fac: pd.DataFrame) -> dict:
    """產業 -> 同時存在於因子與報酬寬表中的股票欄位。"""
    avail = set(fac.columns) & set(ctx.fwd.columns)
    gcols = {}
    for sid, g in ctx.group_map.items():
        if sid in avail:
            gcols.setdefault(g, []).append(sid)
    return gcols


def monthly_ic(ctx: Context, fac: pd.DataFrame, months) -> pd.Series:
    """產業內 Spearman rank IC，跨產業平均（口徑對齊前置專案 factor_eval）。"""
    recs = {}
    gcols = group_cols(ctx, fac)
    for ym in months:
        if ym not in fac.index:
            continue
        frow, rrow = fac.loc[ym], ctx.fwd.loc[ym]
        ics = []
        for g, cols in gcols.items():
            f = frow[cols].astype(float)
            r = rrow[cols].astype(float)
            ok = f.notna() & r.notna()
            if ok.sum() < MIN_STOCKS:
                continue
            fv, rv = f[ok], r[ok]
            if fv.nunique() < 2 or rv.nunique() < 2:
                continue
            ic = fv.rank().corr(rv.rank())
            if pd.notna(ic):
                ics.append(ic)
        if ics:
            recs[ym] = float(np.mean(ics))
    return pd.Series(recs)


def icir(ic: pd.Series):
    if len(ic) < 6 or ic.std() == 0:
        return None, None
    return float(ic.mean()), float(ic.mean() / ic.std())


def xsec_corr(a: pd.DataFrame, b: pd.DataFrame, months, min_n=30) -> float | None:
    """兩因子逐月截面 Spearman 的時間平均（帶號）。"""
    rhos = []
    for ym in months:
        if ym not in a.index or ym not in b.index:
            continue
        x, y = a.loc[ym].astype(float), b.loc[ym].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < min_n:
            continue
        xi, yi = x[ok], y[ok]
        # 任一邊該月為常數 → rank 全同分 → 標準差 0 → pandas 的 .corr() 內部
        # 走 np.corrcoef 會除以零，噴 RuntimeWarning: invalid value in divide。
        # 結果本來就是 NaN 會被下面濾掉，這裡先擋只是為了不洗版。
        # （monthly_ic / industry_ic_series 已有同樣的保護，此處原本漏掉。）
        if xi.nunique() < 2 or yi.nunique() < 2:
            continue
        rho = xi.rank().corr(yi.rank())
        if pd.notna(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else None


def leg_stats(ctx: Context, fac: pd.DataFrame, months, q=0.2,
              cols=None, min_n=30):
    """多空腿：每月因子前/後 20% 等權，相對均值的年化超額。cols 限定範圍。"""
    tops, bots, turn, short_turn = [], [], [], []
    prev_top = prev_bot = None
    for ym in months:
        if ym not in fac.index:
            continue
        f, r = fac.loc[ym], ctx.fwd.loc[ym]
        if cols is not None:
            f, r = f[cols], r[cols]
        ok = f.notna() & r.notna()
        if ok.sum() < min_n:
            continue
        fv, rv = f[ok], r[ok]
        k = max(5, int(len(fv) * q))
        top = set(fv.nlargest(k).index)
        bot = set(fv.nsmallest(k).index)
        univ = rv.mean()
        tops.append(rv[list(top)].mean() - univ)
        bots.append(univ - rv[list(bot)].mean())
        if prev_top is not None:
            turn.append(1 - len(top & prev_top) / max(len(top), 1))
            short_turn.append(1 - len(bot & prev_bot) / max(len(bot), 1))
        prev_top, prev_bot = top, bot
    out = {
        "long_excess_ann": float(np.mean(tops) * 12) if tops else None,
        "short_excess_ann": float(np.mean(bots) * 12) if bots else None,
        "short_turnover_m": float(np.mean(short_turn)) if short_turn else None,
    }
    return out, (float(np.mean(turn)) if turn else None)


def coverage(ctx: Context, fac: pd.DataFrame, months, cols=None) -> float | None:
    fr = []
    for ym in months:
        if ym not in fac.index:
            continue
        frow = fac.loc[ym] if cols is None else fac.loc[ym][cols]
        rrow = ctx.fwd.loc[ym] if cols is None else ctx.fwd.loc[ym][cols]
        have_ret = rrow.notna()
        if have_ret.sum() == 0:
            continue
        fr.append((frow.notna() & have_ret).sum() / have_ret.sum())
    return float(np.mean(fr)) if fr else None


def industry_ic_series(ctx: Context, fac: pd.DataFrame, months) -> dict:
    """{產業: 該產業逐月 IC 序列}。"""
    gcols = group_cols(ctx, fac)
    out = {}
    for g, cols in gcols.items():
        recs = {}
        for ym in months:
            if ym not in fac.index:
                continue
            f = fac.loc[ym][cols].astype(float)
            r = ctx.fwd.loc[ym][cols].astype(float)
            ok = f.notna() & r.notna()
            if ok.sum() < MIN_STOCKS or f[ok].nunique() < 2 or r[ok].nunique() < 2:
                continue
            v = f[ok].rank().corr(r[ok].rank())
            if pd.notna(v):
                recs[ym] = float(v)
        out[g] = pd.Series(recs)
    return out


def industry_qualify(ctx: Context, fac: pd.DataFrame, ind_series: dict) -> list:
    """
    Stage 4b：整體未達標時，檢查是否有單一產業達到（更嚴的）產業專屬門檻。
    回傳 [(產業, train_icir, valid_icir, decay), ...]，依 train_icir 由高到低。
    """
    s4b = FUN.get("stage4b_industry") or {}
    if not s4b.get("enabled"):
        return []
    m_tr, m_va = set(ctx.mask("sub_train")), set(ctx.mask("validation"))
    gcols = group_cols(ctx, fac)
    out = []
    for g, ser in ind_series.items():
        tr = ser[[m for m in ser.index if m in m_tr]]
        va = ser[[m for m in ser.index if m in m_va]]
        if len(tr) < s4b["min_months_train"] or len(va) < s4b["min_months_valid"]:
            continue
        if tr.std() == 0 or va.std() == 0:
            continue
        icir_tr = float(tr.mean() / tr.std())
        icir_va = float(va.mean() / va.std())
        if icir_tr < s4b["min_subtrain_icir"] or icir_va < s4b["min_validation_icir"]:
            continue
        decay = (1 - icir_va / icir_tr) * 100
        if decay > s4b["max_validation_decay_pct"]:
            continue
        cols = gcols.get(g, [])
        legs, _ = leg_stats(ctx, fac, sorted(m_tr | m_va), cols=cols, min_n=10)
        use = trading_use(legs)
        if use == "unusable" or (use == "short_only" and not
                                 FUN["stage4"].get("allow_short_only", False)):
            continue
        if use == "short_only":
            turn = legs["short_turnover_m"]
            if turn is None or turn > FUN["stage4"]["max_monthly_turnover"]:
                continue
        cov = coverage(ctx, fac, ctx.mask("sub_train") + ctx.mask("validation"),
                       cols=cols)
        if (cov or 0) < s4b["min_coverage"]:
            continue
        out.append((g, icir_tr, icir_va, decay))
    return sorted(out, key=lambda t: -t[1])


# ---------------------------------------------------------------------------
# 診斷組裝
# ---------------------------------------------------------------------------

def diagnostics(ctx: Context, fac: pd.DataFrame):
    """一次算齊 sub-train + validation 的全部診斷素材。"""
    m_tr, m_va = ctx.mask("sub_train"), ctx.mask("validation")
    ic_tr = monthly_ic(ctx, fac, m_tr)
    ic_va = monthly_ic(ctx, fac, m_va)
    mean_tr, icir_tr = icir(ic_tr)
    mean_va, icir_va = icir(ic_va)
    decay = None
    if icir_tr and icir_va is not None and icir_tr != 0:
        decay = (1 - icir_va / icir_tr) * 100 if icir_tr > 0 else None

    ic_all = pd.concat([ic_tr, ic_va])
    by_year = {}
    for ym, v in ic_all.items():
        by_year.setdefault(ym[:4], []).append(v)
    ic_by_year = {
        y: [_r(np.mean(v)), ctx.regime.get(y, {}).get("regime", "?")]
        for y, v in sorted(by_year.items())}

    up = [ym for ym in ic_all.index if ctx.mkt.get(ym, np.nan) > 0]
    dn = [ym for ym in ic_all.index if ctx.mkt.get(ym, np.nan) < 0]
    cond = {"up_months": _r(ic_all[ic_all.index.isin(up)].mean()),
            "down_months": _r(ic_all[ic_all.index.isin(dn)].mean())}

    # 產業別 ICIR（sub-train+validation；序列另存供 Stage 4b 使用）
    ind_series = industry_ic_series(ctx, fac, m_tr + m_va)
    ind = {}
    for g, s in ind_series.items():
        if len(s) >= 12 and s.std() > 0:
            ind[g] = _r(s.mean() / s.std(), 2)

    legs, turn = leg_stats(ctx, fac, m_tr + m_va)
    cov = coverage(ctx, fac, m_tr + m_va)

    return {
        "ic_tr": ic_tr, "ic_va": ic_va,
        "sub_train": {"mean_ic": mean_tr, "icir": icir_tr},
        "validation": {"mean_ic": mean_va, "icir": icir_va,
                       "decay_pct": decay},
        "ic_by_year": ic_by_year, "cond_ic": cond,
        "industry_icir": ind, "legs": legs,
        "coverage": cov, "turnover_m": turn,
        "_ind_series": ind_series,   # 內部用（Stage 4b），不進 LLM 輸出
    }


# ---------------------------------------------------------------------------
# 四階段漏斗
# ---------------------------------------------------------------------------

def evaluate_batch(cands: list[dict], ctx: Context) -> list[dict]:
    ids = [c.get("id") for c in cands]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("候選 id 必須是非空且唯一的字串")
    out = {}
    alive = []          # (cand, parsed, fac, diag)

    for c in cands:
        rec = {"id": c["id"], "formula": c.get("formula", ""),
               "groups_seen": sorted(set(ctx.group_map.values()))}
        want = c.get("direction", "pos")
        if want not in ("pos", "neg"):
            rec.update(verdict="rejected_syntax", reason="direction 只能是 pos 或 neg")
            out[c["id"]] = rec
            continue
        # Stage 0
        try:
            pf = dsl.parse(c["formula"], allowed_fields=ctx.fields,
                           window_whitelist=frozenset(CFG["dsl"]["window_whitelist"]),
                           max_depth=CFG["dsl"]["max_depth"],
                           max_fields=CFG["dsl"]["max_fields"])
        except dsl.DSLError as e:
            rec.update(verdict="rejected_syntax", reason=str(e))
            out[c["id"]] = rec
            continue
        rec["fhash"] = pf.fhash
        # 公式級去重（等價哈希撞既有因子庫）
        dup = next((fid for fid, meta in ctx.lib_meta.items()
                    if meta.get("fhash") == pf.fhash), None)
        if dup:
            rec.update(verdict="rejected_duplicate", reason=f"與 {dup} 代數等價")
            out[c["id"]] = rec
            continue

        fac = dsl.Engine(ctx.data, ctx.group_map).eval(pf.tree)
        rec["value_orientation"] = -1 if want == "neg" else 1
        fac = orient_factor(fac, rec)
        diag = diagnostics(ctx, fac)
        rec.update({k: diag[k] for k in
                    ("sub_train", "validation", "ic_by_year", "cond_ic",
                     "industry_icir", "legs", "coverage", "turnover_m")})

        # Stage 1：快速 IC + 方向一致性
        oriented_mean = diag["sub_train"]["mean_ic"]
        mean_tr = None if oriented_mean is None else oriented_mean * rec["value_orientation"]
        rec["raw_mean_ic"] = mean_tr
        if mean_tr is None or abs(mean_tr) < FUN["stage1_min_abs_ic"]:
            rec.update(verdict="rejected_stage1", reason="sub-train IC 低於門檻")
            out[c["id"]] = rec
            continue
        if FUN["stage1_check_direction"] and (
                (want == "pos") != (mean_tr > 0)):
            rec.update(verdict="rejected_stage1",
                       reason=f"IC 方向({'+' if mean_tr > 0 else '-'})與假設宣稱({want})相反"
                              "——假設與證據不一致，可提出反向假設重新提交")
            out[c["id"]] = rec
            continue

        # Stage 2：因子庫去相關（產業限定因子只在其產業範圍內比較）
        if ctx.lib_values:
            worst, worst_rho = None, 0.0
            gc_all = group_cols(ctx, fac)
            for fid, lv in ctx.lib_values.items():
                scope = approved_groups(ctx.lib_meta.get(fid, {}))
                if scope is not None:
                    cols = [s for g in scope for s in gc_all.get(g, [])]
                    if len(cols) < 10:
                        continue
                    rho = xsec_corr(fac[cols], lv[cols],
                                    ctx.mask("sub_train"), min_n=10)
                else:
                    rho = xsec_corr(fac, lv, ctx.mask("sub_train"))
                if rho is not None and abs(rho) > abs(worst_rho):
                    worst, worst_rho = fid, rho
            if worst and abs(worst_rho) > FUN["stage2_max_corr_library"]:
                wmeta = ctx.lib_meta.get(worst, {})
                rec["stage2_culprit"] = {
                    "factor": worst, "rho": _r(worst_rho, 2),
                    "factor_name": wmeta.get("name_zh", ""),
                    "factor_desc": wmeta.get("desc_zh", "")}
                rec.update(verdict="rejected_stage2",
                           reason=f"與庫內 {worst} |ρ|={abs(worst_rho):.2f} > "
                                  f"{FUN['stage2_max_corr_library']}")
                out[c["id"]] = rec
                continue
            if worst:
                rec["stage2_max_rho"] = {"factor": worst, "rho": _r(worst_rho, 2)}

        alive.append((c, pf, fac, diag, rec))

    # Stage 3：批內去重（依 sub-train ICIR 排序，貪婪保留）
    alive.sort(key=lambda t: -(t[3]["sub_train"]["icir"] or -9))
    kept = []
    for c, pf, fac, diag, rec in alive:
        clash = None
        for kc, kpf, kfac, kdiag, krec in kept:
            rho = xsec_corr(fac, kfac, ctx.mask("sub_train"))
            if rho is not None and abs(rho) > FUN["stage3_max_corr_batch"]:
                clash = (kc["id"], rho)
                break
        if clash:
            rec.update(verdict="rejected_stage3",
                       reason=f"與同批 {clash[0]} |ρ|={abs(clash[1]):.2f}，"
                              f"保留 ICIR 較高者")
            out[c["id"]] = rec
        else:
            kept.append((c, pf, fac, diag, rec))

    # Stage 4：完整驗證
    s4 = FUN["stage4"]
    for c, pf, fac, diag, rec in kept:
        fails = []
        st, va = diag["sub_train"], diag["validation"]
        if (st["icir"] or -9) < s4["min_subtrain_icir"]:
            fails.append(f"sub-train ICIR {st['icir']} < {s4['min_subtrain_icir']}")
        if (va["icir"] or -9) < s4["min_validation_icir"]:
            fails.append(f"validation ICIR {va['icir']} < {s4['min_validation_icir']}")
        if va["decay_pct"] is not None and va["decay_pct"] > s4["max_validation_decay_pct"]:
            fails.append(f"validation 衰減 {va['decay_pct']:.0f}% > "
                         f"{s4['max_validation_decay_pct']}%（過擬合徵兆）")
        use = trading_use(diag["legs"])
        rec["trading_use"] = use
        if s4["require_long_leg_positive"] and (
                use not in ("long_only", "long_short") and not
                (use == "short_only" and s4.get("allow_short_only", False))):
            fails.append("沒有符合設定的正超額交易腿")
        if (diag["coverage"] or 0) < s4["min_coverage"]:
            fails.append(f"覆蓋率 {diag['coverage']} < {s4['min_coverage']}")
        turn = (diag["legs"].get("short_turnover_m") if use == "short_only"
                else diag["turnover_m"])
        rec["turnover_m"] = turn
        if turn is None or not np.isfinite(turn) or turn > s4["max_monthly_turnover"]:
            fails.append(f"交易腿月換手 {turn} 缺值或 > {s4['max_monthly_turnover']}")
        if not fails:
            rec.update(verdict="passed")
        else:
            # Stage 4b：整體未達標 → 檢查產業專屬通道（門檻更嚴）
            quals = industry_qualify(ctx, fac, diag["_ind_series"])
            if quals:
                g, itr, iva, dec = quals[0]
                rec["industry_scope"] = g
                rec["industry_metrics"] = {"train_icir": itr,
                                           "valid_icir": iva, "decay_pct": dec}
                legs, turn = leg_stats(ctx, fac,
                    ctx.mask("sub_train") + ctx.mask("validation"),
                    cols=group_cols(ctx, fac)[g], min_n=10)
                rec["trading_use"] = trading_use(legs)
                rec["industry_metrics"]["legs"] = legs
                rec["turnover_m"] = (legs["short_turnover_m"]
                    if rec["trading_use"] == "short_only" else turn)
                rec.update(verdict="passed_industry",
                           reason=f"整體未達標（{'; '.join(fails)}），"
                                  f"但於「{g}」產業內達到專屬門檻")
            else:
                rec.update(verdict="rejected_stage4", reason="; ".join(fails))
        out[c["id"]] = rec

    return [out[c["id"]] for c in cands]


def sealed_test_metrics(ctx: Context, fac: pd.DataFrame,
                        subtrain_icir: float | None) -> dict:
    """
    ⛔ test 期指標——只允許被 memory.admit() 寫進 library.json 的密封欄位。
    呼叫方嚴禁 print 或放入任何回傳給 LLM 的結構（規格 9.1）。
    """
    ic_te = monthly_ic(ctx, fac, ctx.mask("test"))
    mean_te, icir_te = icir(ic_te)
    decay = None
    if subtrain_icir and icir_te is not None and subtrain_icir > 0:
        decay = (1 - icir_te / subtrain_icir) * 100
    return {"mean_ic": _r(mean_te), "icir": _r(icir_te, 2),
            "decay_vs_subtrain_pct": _r(decay, 0), "n_months": len(ic_te)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def self_test(ctx: Context):
    batch = [
        {"id": "T-01", "formula": "cs_rank(rev_yoy_ma3) * sign(mom_60)",
         "category": "revenue_momentum", "direction": "pos",
         "hypothesis": "營收動能+價格確認", "prediction": "ICIR>0.4"},
        {"id": "T-02", "formula": "neg(cs_rank(vol_126))",
         "category": "low_vol", "direction": "pos",
         "hypothesis": "低波動溢酬", "prediction": "ICIR>0.3"},
        {"id": "T-03", "formula": "ts_mean(f_bad, 6)",
         "category": "x", "direction": "pos", "hypothesis": "x", "prediction": "x"},
        {"id": "T-04", "formula": "cs_rank(rev_yoy_ma3 + rev_yoy)",
         "category": "revenue_momentum", "direction": "pos",
         "hypothesis": "與 T-01 高相關，應死於 Stage 3", "prediction": "-"},
    ]
    res = evaluate_batch(batch, ctx)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", dest="outp")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    ctx = Context()
    if a.self_test:
        self_test(ctx)
        return
    cands = json.loads(Path(a.inp).read_text(encoding="utf-8"))
    res = evaluate_batch(cands, ctx)
    Path(a.outp).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    n_pass = sum(1 for r in res if r["verdict"] == "passed")
    print(f"✅ {len(res)} 候選 → {n_pass} passed → {a.outp}")


if __name__ == "__main__":
    main()
