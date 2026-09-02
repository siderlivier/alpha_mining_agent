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
    dmap = {d["id"]: d for d in diags}
    admitted = []
    for c in cands:
        d = dmap.get(c["id"])
        if not d or d.get("verdict") not in ("passed", "passed_industry"):
            continue
        pf = dsl.parse(c["formula"], allowed_fields=ctx.fields)
        fac = dsl.Engine(ctx.data, ctx.group_map).eval(pf.tree)
        scope = d.get("industry_scope")
        if scope:
            # 產業限定：密封 test 指標只在該產業內計算
            cols = ec.group_cols(ctx, fac).get(scope, [])
            fac_s = fac[cols]
            base_icir = (d.get("industry_metrics") or {}).get("train_icir")
            sealed = ec.sealed_test_metrics(ctx, fac_s, base_icir)
        else:
            sealed = ec.sealed_test_metrics(
                ctx, fac, (d.get("sub_train") or {}).get("icir"))
        fid = mem.admit(c, d, pf, fac, test_metrics_sealed=sealed,
                        round_id=round_id, industry_scope=scope)
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
