"""Reproduce upstream DFS comparisons, with a separate signal-first audit.

No formal library writes. The upstream-compatible table retains upstream's
future-return filtering and is explicitly NOT a leakage-free performance claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import factor_lab as fl
from memory import Memory, approved_groups
ROOT = Path(__file__).resolve().parents[1]
UP = ROOT.parent / "tw_alpha_strategy"
ARCHIVE = ROOT / "data/rebuild_history/20260919_234830_priority_repairs/before/memory/library.json"
STRATEGY = ["gp_to_px", "fcf_yield", "bp", "quality_score", "neg_accruals", "neg_vol_63", "dist_high", "g_eps_12"]


def liquidity_mask(frame):
    return frame.groupby(["ym", "group"]).amt_21.rank(pct=True).ge(.70)


def buffered_ids(frame, previous, top_q=.10, exit_q=.20):
    ranked = frame.sort_values(["score", "stock_id"], ascending=[False, True])
    n = max(1, int(len(ranked)*top_q))
    pct = (np.arange(len(ranked))+1)/len(ranked)
    kept = [s for s,p in zip(ranked.stock_id,pct) if s in previous and p <= exit_q]
    # User's explicit retention rule: do not force-sell an in-buffer holding
    # merely because the eligible universe shrank this month.
    entries = [s for s in ranked.head(n).stock_id if s not in kept]
    return kept + entries[:max(0,n-len(kept))]


def signal_audit(frame):
    previous = {}
    rows = []
    holdings = []
    for ym, month in frame.groupby("ym", sort=True):
        old_ids = set().union(*previous.values()) if previous else set()
        planned = []
        next_previous = {}
        for group, sub in month.dropna(subset=["score"]).groupby("group"):
            if len(sub)<8:continue
            ids = buffered_ids(sub, previous.get(group,set()))
            intended = sub[sub.stock_id.isin(ids)]
            # t+1 fill status may prevent a new purchase, but may not choose
            # a substitute using future return availability.
            held = previous.get(group,set())
            filled = intended[intended.stock_id.isin(held) | intended.buyable.eq(True)]
            next_previous[group] = set(filled.stock_id)
            planned.append(filled)
        previous = next_previous
        if not planned:continue
        selected = pd.concat(planned)
        bad = selected[selected.r_vwap.isna()]
        current_ids = set(selected.stock_id)
        union = old_ids | current_ids
        turnover = len(old_ids ^ current_ids)/len(union) if union else 0.
        net_return = float(selected.r_vwap.mean()-.004*turnover) if len(selected) and bad.empty else None
        rows.append(dict(ym=ym, selected=len(selected), missing_returns=len(bad),
                         unknown_ids=bad.stock_id.tolist(),turnover=turnover,net_return=net_return))
        holdings.extend(dict(ym=ym,stock_id=s) for s in selected.stock_id)
    return rows, pd.DataFrame(holdings)


def score_panel(frame, feats, kind):
    out=frame.copy()
    g=out.groupby(["ym","group"])
    for c in feats:
        out[c]=(out[c]-g[c].transform("mean"))/g[c].transform("std").replace(0,np.nan)
    out[feats]=out[feats].fillna(0.)
    out["y"]=out.fwd_ret_1m-g.fwd_ret_1m.transform("mean")
    pred=pd.Series(np.nan,index=out.index)
    for cut,months in fl.segments(sorted(out.ym.unique())):
        tr=out[out.ym.le(cut)&out.y.notna()]
        te=out[out.ym.isin(months)]
        if len(tr)>=1000 and len(te):
            pred.loc[te.index]=fl._fit_predict(kind,tr[feats].to_numpy(),tr.y.to_numpy(),te[feats].to_numpy())
    return pred


def stats(r):
    if len(r)<6:return {}
    x=r.long
    curve=(1+x).cumprod()
    ex=x-r.benchmark
    return dict(months=len(r),CAGR=float(curve.iloc[-1]**(12/len(x))-1),
        Sharpe=float(x.mean()/x.std()*np.sqrt(12)),
        MaxDD=float((curve/curve.cummax().clip(lower=1)-1).min()),
        Excess=float(ex.mean()*12),IR=float(ex.mean()/ex.std()*np.sqrt(12)),
        turnover=float(r.turnover.mean()),holdings=float(r.nhold.mean()),
        BenchCAGR=float((1+r.benchmark).prod()**(12/len(r))-1))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--universe",choices=["upstream","common"],default="upstream")
    a=ap.parse_args()
    out=a.output or ROOT/"logs"/("dfs_comparison_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():
        raise SystemExit("Output already contains a run; choose a new directory")
    (out/"runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    # Isolated subprocess prevents upstream config/backtest names from shadowing
    # our local modules. It only reads the fresh upstream cache and exports locally.
    code="""import sys,pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
up=Path(sys.argv[1]);sys.path.insert(0,str(up/'src'))
import mine_dfs
cache=up/'data/processed/_monthly_base.parquet'
assert cache.stat().st_mtime >= (up/'data/processed/panel.parquet').stat().st_mtime, 'upstream monthly cache stale'
m=pd.read_parquet(cache);d,names=mine_dfs.generate(m,exec_mode='close_t0')
d.to_parquet(sys.argv[2],index=False)
"""
    inputs=[ROOT/"data/monthly_base.parquet",ROOT/"memory/library.json",ROOT/"memory/factor_values.parquet",ROOT/"config.yaml",ROOT/"src/factor_lab.py",Path(__file__),ARCHIVE]
    inputs += [UP/"src"/n for n in ["mine_dfs.py","backtest.py","config.py","exec_mode.py","ml/main_strategy.py","ml/ml_model.py","ml/preprocess.py"]]
    inputs += [UP/"data/processed"/n for n in ["panel.parquet","_monthly_base.parquet","exec_prices.parquet"]]
    event=UP/"data/raw/event_parvalue.parquet"
    if event.exists():inputs.append(event)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest=dict(status="running",inputs=hashes,liquidity=.30,scope="group",entry=.10,exit=.20,
        execution="vwap_t1",cost=.004,weighting="equal",smooth=1.,
        universe=a.universe,warning="Compatibility results inherit future-label filtering; see signal-first audit.")
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    subprocess.run([sys.executable,"-B","-c",code,str(UP),str(out/"dfs_generated.parquet")],check=True)
    dfs=pd.read_parquet(out/"dfs_generated.parquet");dfs.ym=dfs.ym.astype(str);dfs.stock_id=dfs.stock_id.astype(str)
    base=pd.read_parquet(ROOT/"data/monthly_base.parquet")
    monthly=pd.read_parquet(UP/"data/processed/_monthly_base.parquet",columns=["stock_id","ym","amt_21"])
    monthly.ym=monthly.ym.astype(str);monthly.stock_id=monthly.stock_id.astype(str)
    px=pd.read_parquet(UP/"data/processed/exec_prices.parquet");px.ym=px.ym.astype(str);px.stock_id=px.stock_id.astype(str)
    lib,values=Memory().snapshot()
    own=sorted(k for k,m in lib.items() if not m.get("reference") and m.get("trading_use")!="short_only")
    wide=values[values.factor_id.isin(own)].pivot(index=["ym","stock_id"],columns="factor_id",values="value").reset_index()
    archive=json.loads(ARCHIVE.read_text(encoding="utf-8"))
    original=[m["dfs_name"] for m in archive.values() if m.get("reference")]
    aliases={"op_margin":"op_to_rev"}
    for n in original:
        if n not in dfs and n in aliases:dfs[n]=dfs[aliases[n]]
    missing=set(original+STRATEGY)-set(dfs)
    if missing:raise ValueError(f"Missing DFS definitions: {missing}")
    cols=list(dict.fromkeys(original+STRATEGY))
    d=base[["stock_id","ym","group","fwd_ret_1m"]].merge(monthly,on=["stock_id","ym"],validate="one_to_one")
    d=d.merge(dfs[["stock_id","ym"]+cols],on=["stock_id","ym"],validate="one_to_one")
    d=d.merge(wide,on=["stock_id","ym"],how="left",validate="one_to_one")
    d=d.merge(px[["stock_id","ym","r_vwap","buyable"]],on=["stock_id","ym"],how="left",validate="one_to_one")
    for fid in own:
        groups=approved_groups(lib[fid])
        if groups is not None:d.loc[~d.group.isin(groups),fid]=np.nan
    # Freeze factor coverage and DFS orientation using sub_train only.
    tr=d.ym.between(*fl.CFG["split"]["sub_train"])
    own=[f for f in own if d.loc[tr,f].notna().mean()>=fl.MIN_COVERAGE]
    orientation={n:(-1 if archive[k].get("icir_train",0)<0 else 1) for k in archive if archive[k].get("reference") for n in [archive[k]["dfs_name"]]}
    # Historical train direction is recorded explicitly. No test-based redirection.
    dfs_oriented=[]
    for n in original:
        c="DFS__"+n;d[c]=d[n]*orientation[n];dfs_oriented.append(c)
    groups={"agent":own,"dfs25":dfs_oriented,"dfs8":STRATEGY,"agent_plus_dfs25":own+dfs_oriented}
    common_eligible=liquidity_mask(d)  # before any return/score availability filtering
    coverage=pd.concat([d[feats].notna().mean(axis=1).ge(.5) for feats in groups.values()],axis=1).all(axis=1)
    common_eligible &= coverage  # common investable universe; never consult future labels
    manifest.update(features=groups,aliases=aliases,dfs_orientation=orientation,
        coverage_policy=("50% per pool, then industry liquidity rank among scorable stocks (upstream ML order)." if a.universe=="upstream" else "At least 50% observed features in EVERY compared pool; common intersection after full-industry liquidity ranking."))
    # Complete monthly execution cycles: panel ends 2026-06-25, so the final
    # available July execution is absent; last complete signal month is April.
    end=str(pd.Period(base.ym.max(),freq="M")-2)
    test=(fl.SPANS["test"][0],min(fl.SPANS["test"][1],end))
    manifest["eligible_by_group"]={}
    result=[]
    for name,feats in groups.items():
        for kind in ["equal","ridge","lgbm"]:
            print("SCORING",name,kind,len(feats),flush=True)
            d["score"]=score_panel(d,feats,kind)
            if a.universe=="upstream":
                scored=d[d.score.notna()&d[feats].notna().mean(axis=1).ge(.5)&d.amt_21.notna()]
                eligible=liquidity_mask(scored).reindex(d.index,fill_value=False)
            else:
                eligible=common_eligible
            sample=d[eligible&d.ym.between(*test)].copy()
            manifest["eligible_by_group"][name+"_"+kind]=sample.groupby("group").size().to_dict()
            sample.to_parquet(out/(name+"_"+kind+"_scored.parquet"),index=False)
            audit,hold=signal_audit(sample)
            (out/(name+"_"+kind+"_audit.json")).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
            hold.to_csv(out/(name+"_"+kind+"_signal_holdings.csv"),index=False)
            # Run the real upstream portfolio code in its own interpreter.
            code2="""import sys,pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
import backtest as b
d=pd.read_parquet(sys.argv[2]);d['fwd_ret_1m']=d.r_vwap.where(d.buyable.eq(True))
r=b.portfolio_returns(d,top_q=.10,weighting='equal',buffer_exit_q=.20,smooth_w=1.)
r.to_csv(sys.argv[3],index_label='ym')
"""
            target=out/(name+"_"+kind+"_monthly.csv")
            subprocess.run([sys.executable,"-B","-c",code2,str(UP),str(out/(name+"_"+kind+"_scored.parquet")),str(target)],check=True)
            r=pd.read_csv(target,index_col="ym")
            row=dict(group=name,model=kind,NFactors=len(feats),**stats(r),signal_missing_months=sum(x["missing_returns"]>0 for x in audit))
            clean=[x["net_return"] for x in audit]
            complete=len(clean)==len(r) and all(x is not None for x in clean)
            row["signal_first_CAGR"] = float(np.prod(1+np.array(clean))**(12/len(clean))-1) if complete else None
            result.append(row)
            (out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
            print("RESULT",json.dumps(row),flush=True)
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, "Inputs changed"
    manifest.update(status="complete",test=test)
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print("OUTPUT",out,flush=True)


if __name__=="__main__":main()
