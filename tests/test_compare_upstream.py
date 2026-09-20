"""Standalone comparison regression tests and saved-run holdings audit.

pytest: python -m pytest tests/test_compare_upstream.py -q
Evidence export: python tests/test_compare_upstream.py --audit-run logs/<run>
"""
import sys
import json
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from compare_upstream import buffered_ids, liquidity_mask, signal_audit


def month(ym="2020-01", n=20):
    return pd.DataFrame(dict(stock_id=[f"S{i:02}" for i in range(n)],
        score=np.arange(n,0,-1,dtype=float),group="A",ym=ym,buyable=True,r_vwap=.01,amt_21=np.arange(n)))


def test_liquidity_ranks_within_each_industry_and_month():
    a=month(n=10); b=month(n=10); b.group="B"; b.amt_21+=1000
    d=pd.concat([a,b],ignore_index=True)
    chosen=d[liquidity_mask(d)]
    # Exactly reproduce upstream percentile >= .70, including boundary ties.
    assert chosen.groupby("group").size().to_dict()=={"A":4,"B":4}


def test_buffer_retains_top_twenty_but_exits_below_it():
    d=month()
    assert buffered_ids(d,{"S03"})==["S03","S00"]
    assert buffered_ids(d,{"S04"})==["S00","S01"]


def test_buffer_does_not_force_sell_when_universe_shrinks():
    assert buffered_ids(month(n=10),{"S00","S01"})==["S00","S01"]


def test_missing_future_return_never_changes_selection_or_silently_fills():
    d=month(); expected=signal_audit(d)[1]
    d.loc[0,"r_vwap"]=np.nan
    audit,hold=signal_audit(d)
    pd.testing.assert_frame_equal(expected,hold)
    assert audit[0]["unknown_ids"]==["S00"] and audit[0]["net_return"] is None


def test_unfilled_entry_does_not_get_future_informed_substitute():
    d=month(); d.loc[0,"buyable"]=False
    audit,hold=signal_audit(d)
    assert hold.stock_id.tolist()==["S01"]


def test_missing_industry_resets_previous_holdings():
    a=month(); b=month("2020-02"); b.group="B"
    c=month("2020-03"); c.loc[0,"score"]=17.5
    _,hold=signal_audit(pd.concat([a,b,c],ignore_index=True))
    assert "S00" not in hold[hold.ym.eq("2020-03")].stock_id.tolist()


def test_future_labels_cannot_change_current_prediction(monkeypatch):
    import compare_upstream as c
    monkeypatch.setattr(c.fl,"segments",lambda _: [("2019-11",["2020-01"])])
    d=pd.concat([month(ym,n=600) for ym in ["2019-10","2019-11","2020-01"]],ignore_index=True)
    d["factor"]=d.score**.5
    d["fwd_ret_1m"]=d.score/6000
    expected=c.score_panel(d,["factor"],"ridge")
    d.loc[d.ym.eq("2020-01"),"fwd_ret_1m"]=np.nan
    actual=c.score_panel(d,["factor"],"ridge")
    pd.testing.assert_series_equal(expected,actual)
    assert actual[d.ym.eq("2020-01")].notna().all()


def audit_saved_run(run):
    """Reconstruct liquidity ranks; independently replay saved signal selection.

    Uses archived scores, never trains a new model or changes the library.
    Refuses changed source data rather than mixing versions.
    """
    import compare_upstream as c
    run=Path(run)
    manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="complete" and manifest["universe"]=="upstream"
    paths=[c.ROOT/"memory/library.json",c.ROOT/"memory/factor_values.parquet",
           c.ROOT/"data/monthly_base.parquet",c.UP/"data/processed/_monthly_base.parquet"]
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest["inputs"][str(p)], f"Changed input: {p}"
    lib=json.loads(paths[0].read_text(encoding="utf-8"))
    own=manifest["features"]["agent"]
    vals=pd.read_parquet(paths[1])
    wide=vals[vals.factor_id.isin(own)].pivot(index=["ym","stock_id"],columns="factor_id",values="value").reset_index()
    base=pd.read_parquet(paths[2],columns=["stock_id","ym","group"])
    monthly=pd.read_parquet(paths[3],columns=["stock_id","ym","amt_21"])
    dfs=pd.read_parquet(run/"dfs_generated.parquet")
    for frame in [base,monthly,wide,dfs]:
        frame.stock_id=frame.stock_id.astype(str);frame.ym=frame.ym.astype(str)
    d=base.merge(monthly,on=["stock_id","ym"],validate="one_to_one").merge(wide,on=["stock_id","ym"],how="left",validate="one_to_one")
    for fid in own:
        scope=c.approved_groups(lib[fid])
        if scope is not None:d.loc[~d.group.isin(scope),fid]=np.nan
    for alias,source in manifest["aliases"].items():dfs[alias]=dfs[source]
    cols=list(dict.fromkeys(list(manifest["dfs_orientation"])+manifest["features"]["dfs8"]))
    d=d.merge(dfs[["stock_id","ym"]+cols],on=["stock_id","ym"],validate="one_to_one")
    for name,sign in manifest["dfs_orientation"].items():d["DFS__"+name]=d[name]*sign
    evidence=[]
    for pool,feats in manifest["features"].items():
        pre=d[d[feats].notna().mean(axis=1).ge(.5)&d.amt_21.notna()].copy()
        pre["liquidity_pct"]=pre.groupby(["ym","group"]).amt_21.rank(pct=True)
        ranks=pre[["stock_id","ym","liquidity_pct"]]
        for kind in ["equal","ridge","lgbm"]:
            tag=pool+"_"+kind
            scored=pd.read_parquet(run/(tag+"_scored.parquet"))
            scored=scored.merge(ranks,on=["stock_id","ym"],validate="one_to_one")
            assert scored.liquidity_pct.ge(.70).all()
            replay,holdings=signal_audit(scored)
            saved=json.loads((run/(tag+"_audit.json")).read_text(encoding="utf-8"))
            assert [(x["ym"],x["unknown_ids"]) for x in replay]==[(x["ym"],x["unknown_ids"]) for x in saved]
            recorded=pd.read_csv(run/(tag+"_signal_holdings.csv"),dtype=str)
            assert set(map(tuple,holdings[["ym","stock_id"]].to_numpy()))==set(map(tuple,recorded[["ym","stock_id"]].to_numpy()))
            previous=set()
            for ym,month_rows in scored.groupby("ym",sort=True):
                for group,sub in month_rows.groupby("group"):
                    ranked=sub.sort_values(["score","stock_id"],ascending=[False,True]).copy()
                    ranked["score_rank"]=np.arange(len(ranked))+1
                    selected=set(holdings.loc[holdings.ym.eq(ym),"stock_id"])
                    bad=ranked[ranked.stock_id.isin(selected)&ranked.r_vwap.isna()]
                    for row in bad.itertuples():
                        retained=row.stock_id in previous
                        assert row.score_rank/len(ranked)<= (.20 if retained else .10) or (not retained and row.score_rank==1 and len(ranked)<10)
                        evidence.append(dict(run=tag,ym=ym,stock_id=row.stock_id,group=group,
                            liquidity_pct=row.liquidity_pct,score_rank=row.score_rank,rank_pool=len(ranked),
                            score_top_fraction=row.score_rank/len(ranked),retained=retained,
                            selected=True,missing_return=True))
                previous=set(holdings.loc[holdings.ym.eq(ym),"stock_id"])
    return pd.DataFrame(evidence)


def test_saved_run_missing_returns_are_selected_after_liquidity_filter():
    import pytest
    run=Path(__file__).resolve().parents[1]/"logs/dfs_comparison_20260920_150651"
    if not (run/"dfs_generated.parquet").exists():
        pytest.skip("Local historical run artifacts are required")
    evidence=audit_saved_run(run)
    assert evidence[["ym","stock_id"]].drop_duplicates().shape[0]==29
    assert evidence.stock_id.nunique()==17
    assert evidence.liquidity_pct.ge(.70).all()


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-run",type=Path,required=True)
    args=ap.parse_args()
    evidence=audit_saved_run(args.audit_run)
    target=args.audit_run/"selection_rank_evidence.csv"
    evidence.to_csv(target,index=False,encoding="utf-8-sig")
    print(evidence[["ym","stock_id"]].drop_duplicates().shape[0],"stock-months;",evidence.stock_id.nunique(),"stocks")
    print(target)
