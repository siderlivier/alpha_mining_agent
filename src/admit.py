"""
M2：入庫工具——把評估通過（passed）的候選寫入因子庫。

流程：讀候選批次 + 診斷結果 → 對每個 passed 且備齊 name_zh/desc_zh 的候選：
  重算因子值 → 計算密封 test 指標（不輸出）→ memory.admit()

stdout 只顯示中繼資料（中文名/面向/公式），密封指標絕不列印。
（M3 的 orchestrator 會直接呼叫 admit_from_results()，本 CLI 供手動操作。）

執行：python src/admit.py --cands batch.json --diags diags.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsl
import eval_candidates as ec
from memory import Memory


def admit_from_results(cands: list[dict], diags: list[dict],
                       ctx: ec.Context | None = None,
                       mem: Memory | None = None,
                       round_id: int | None = None) -> list[str]:
    ctx = ctx or ec.Context()
    mem = mem or Memory()
    mem.ensure()
    cand_ids = [c["id"] for c in cands]
    diag_ids = [d["id"] for d in diags]
    if len(set(cand_ids)) != len(cand_ids) or len(set(diag_ids)) != len(diag_ids):
        raise ValueError("入庫候選與診斷 id 各自必須唯一")
    dmap = {d["id"]: d for d in diags}
    admitted = []
    for c in cands:
        d = dmap.get(c["id"])
        if not d or d.get("verdict") not in ("passed", "passed_industry"):
            continue
        if d.get("formula", c["formula"]) != c["formula"]:
            raise ValueError(f"{c['id']} 公式與診斷不一致，請重新評估")
        expected_orientation = -1 if c.get("direction", "pos") == "neg" else 1
        if d.get("value_orientation", 1) != expected_orientation:
            raise ValueError(f"{c['id']} 方向與診斷不一致，請重新評估")
        pf = dsl.parse(c["formula"], allowed_fields=ctx.fields)
        fac = dsl.Engine(ctx.data, ctx.group_map).eval(pf.tree)
        fac = ec.orient_factor(fac, d)
        scope = d.get("industry_scope")
        if scope:
            # 產業限定：密封 test 指標只在該產業內計算
            scopes = [scope] if isinstance(scope, str) else scope
            cols = [s for g in scopes for s in ec.group_cols(ctx, fac).get(g, [])]
            fac_s = fac[cols]
            base_icir = (d.get("industry_metrics") or {}).get("train_icir")
            sealed = ec.sealed_test_metrics(ctx, fac_s, base_icir)
            fac = fac_s
        else:
            sealed = ec.sealed_test_metrics(
                ctx, fac, (d.get("sub_train") or {}).get("icir"))
        fid = mem.admit(c, d, pf, fac, test_metrics_sealed=sealed,
                        round_id=round_id, industry_scope=scope, group_map=ctx.group_map)
        meta = mem.library()[fid]
        tag = f"｜{scope}限定" if scope else ""
        print(f"✅ {fid}【{meta['name_zh']}】({meta['aspect']}/{meta['category']}{tag}) "
              f"{meta['formula']}（test 指標已密封）")
        admitted.append(fid)
    return admitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--diags", required=True)
    ap.add_argument("--round", type=int, default=None)
    a = ap.parse_args()
    cands = json.loads(Path(a.cands).read_text(encoding="utf-8"))
    diags = json.loads(Path(a.diags).read_text(encoding="utf-8"))
    ids = admit_from_results(cands, diags, round_id=a.round)
    print(f"共入庫 {len(ids)} 個因子")


if __name__ == "__main__":
    main()
