import json
import numpy as np
import pandas as pd
import pytest
import build_base as bb
import seed_reference as sr
from memory import Memory


def test_calendar_gap_and_stock_isolation():
    d = pd.DataFrame(dict(stock_id=["A","A","A","B"],ym=["2019-12","2020-02","2020-03","2020-01"],close=[10.,20.,30.,100.]))
    got = bb.calendar_value(d,"close",1)
    assert pd.isna(got[0]) and got[1] == 30 and pd.isna(got[2])
    assert pd.isna(bb.calendar_value(d,"close",-12)).all()


def test_actual_shares_ignore_capital_and_invalid_values():
    d = pd.DataFrame(dict(shares_issued=[100.,400.,0.,-1.,np.inf,np.nan],b_CapitalStock=[1000.]*6))
    got = bb.issued_shares(d)
    assert got.iloc[:2].tolist()==[100.,400.]
    assert got.iloc[2:].isna().all()
    assert (1000 / got.iloc[:2]).tolist()==[10.,2.5]


def test_transaction_failure_restores_both(tmp_path, monkeypatch):
    import memory
    m=Memory(tmp_path);m.ensure()
    values=pd.DataFrame(dict(factor_id=["F-001"],ym=["2015-01"],stock_id=["A"],value=[1.]))
    m.commit_snapshot({"F-001":{}},values)
    original=(m.lib_path.read_bytes(),m.values_path.read_bytes())
    replace=memory.os.replace
    def fail(source,target):
        if str(source).endswith("library.new"):
            raise OSError("injected failure")
        return replace(source,target)
    monkeypatch.setattr(memory.os,"replace",fail)
    with pytest.raises(OSError):m.commit_snapshot({"F-001":{"changed":True}},values.assign(value=2.))
    assert original==(m.lib_path.read_bytes(),m.values_path.read_bytes())
    assert not (tmp_path/".memory_pending.json").exists()


def test_reference_selection_ignores_test_and_accepts_negative_direction():
    d=pd.DataFrame(dict(factor=["x"],ICIR_train=[-.8],ICIR_validation=[-.6],t_train=[-4.],n_train=[60],n_validation=[20],ICIR_test=[100.]))
    assert sr.select(d,.3,50).factor.tolist()==["x"]
    d.ICIR_test=-100
    assert sr.select(d,.3,50).factor.tolist()==["x"]
    d.ICIR_validation=.6
    assert sr.select(d,.3,50).empty


def test_reference_dedupe_ignores_test():
    pre=pd.DataFrame(dict(dfs_name=["a","a","b","b"],ym=["2015-01"]*4,stock_id=["x","y"]*2,value=[1.,2.,2.,4.]))
    post=pre.assign(ym="2021-01",value=[1.,2.,4.,2.])
    row=pd.DataFrame(dict(ICIR_train=[.9,.8]),index=["a","b"])
    assert sr.dedupe_reference(pre,["a","b"],row,.95)==sr.dedupe_reference(pd.concat([pre,post]),["a","b"],row,.95)


def test_mock_round_refuses_production_memory():
    import mining_loop as ml
    with pytest.raises(ValueError,match="production"):
        ml.run_round(None,Memory(),1,ml.Meter(),lambda p:"[]")


def test_reference_scores_do_not_read_test(tmp_path, monkeypatch):
    (tmp_path/"fields.yaml").write_text("quality: {x: {desc: example}}",encoding="utf-8")
    monkeypatch.setattr(sr,"ROOT",tmp_path)
    rng=np.random.default_rng(13)
    months=pd.period_range("2012-01",periods=110,freq="M").astype(str)
    b=pd.DataFrame([(m,str(i),"g") for m in months for i in range(20)],columns=["ym","stock_id","group"])
    b["x"]=rng.normal(size=len(b));b["fwd_ret_1m"]=b.x*.03+rng.normal(0,.05,len(b))
    before,_=sr.reference_candidates(b)
    future=b.ym.ge("2020-01")
    b.loc[future,"x"]*= -100
    b.loc[future,"fwd_ret_1m"]=np.nan
    after,_=sr.reference_candidates(b)
    pd.testing.assert_frame_equal(before,after)


def test_interrupted_transaction_recovered_before_read(tmp_path):
    m=Memory(tmp_path);m.ensure()
    vals=pd.DataFrame(dict(factor_id=["F-001"],ym=["2015-01"],stock_id=["A"],value=[1.]))
    m.commit_snapshot({"F-001":{}},vals)
    original=m.lib_path.read_bytes(),m.values_path.read_bytes()
    import shutil
    name="a"*32;folder=tmp_path/"transactions"/name;folder.mkdir()
    shutil.copy2(m.lib_path,folder/"library.before");shutil.copy2(m.values_path,folder/"values.before")
    m.lib_path.write_text("{}")
    (tmp_path/".memory_pending.json").write_text(json.dumps(dict(transaction=name,existed=dict(library=True,values=True))))
    lib,v=m.snapshot()
    assert "F-001" in lib
    assert original==(m.lib_path.read_bytes(),m.values_path.read_bytes())


@pytest.mark.parametrize("value", [
    [{"text": "right ] and left [ with ```json content"}],
    [{"text": 'escaped quote " and backslash \\', "nested": [[1,2],[]]}],
    [],
])
def test_llm_array_preserves_string_brackets_and_fences(value):
    import mining_loop as ml
    assert ml.parse_json_array("```json\n"+json.dumps(value)+"\n``` trailing")==value


def test_malformed_outer_array_is_not_salvaged():
    import mining_loop as ml
    with pytest.raises(ValueError): ml.parse_json_array('[{"x": [1,2]}')


@pytest.mark.parametrize("code,is_error", [(1,False),(0,True),(0,False)])
def test_cli_usage_counted_once_on_success_and_failure(monkeypatch,code,is_error):
    import mining_loop as ml
    from types import SimpleNamespace
    monkeypatch.setattr(ml.shutil,"which",lambda x:"cli.exe")
    monkeypatch.setitem(ml.BUD,"weekly_token_budget",0)
    monkeypatch.setitem(ml.BUD,"weekly_cost_budget_usd",0)
    payload=dict(result="[]",is_error=is_error,usage=dict(input_tokens=10,output_tokens=7,cache_read_input_tokens=3),total_cost_usd=.12)
    monkeypatch.setattr(ml.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=code,stdout=json.dumps(payload),stderr=""))
    meter=ml.Meter()
    if code or is_error:
        with pytest.raises(RuntimeError): ml.call_llm("prompt",meter=meter)
    else: ml.call_llm("prompt",meter=meter)
    assert (meter.calls,meter.tokens,meter.cost,meter.estimated_calls,meter.unknown_cost_calls)==(1,20,.12,0,0)
    b=ml.commit_usage({},meter)
    assert b["tokens_used"]==20 and b["cost_usd"]==.12


def test_timeout_retains_returned_usage(monkeypatch,tmp_path):
    import mining_loop as ml
    monkeypatch.setattr(ml.shutil,"which",lambda x:"cli.exe")
    monkeypatch.setitem(ml.BUD,"weekly_token_budget",0)
    monkeypatch.setitem(ml.BUD,"weekly_cost_budget_usd",0)
    monkeypatch.setattr(ml,"dump_raw",lambda *a:tmp_path/"timeout.txt")
    payload=json.dumps(dict(error="timeout",usage=dict(input_tokens=30,output_tokens=9),total_cost_usd=.2))
    def timeout(*a,**k):raise ml.subprocess.TimeoutExpired("cli",2,output=payload.encode())
    monkeypatch.setattr(ml.subprocess,"run",timeout)
    meter=ml.Meter()
    with pytest.raises(RuntimeError):ml.call_llm("prompt",meter=meter)
    assert meter.tokens==39 and meter.calls==1 and meter.timeouts==1 and meter.cost==.2


def test_unknown_cost_is_not_reported_as_free(monkeypatch):
    import mining_loop as ml
    r=ml._parse_envelope('{"error":"failed","usage":{"input_tokens":3}}',"prompt")
    m=ml.Meter();m.add(r);b=ml._blank_budget(0);ml.commit_usage(b,m)
    assert m.tokens==3 and b["unknown_cost_calls"]==1
    monkeypatch.setitem(ml.BUD,"weekly_cost_budget_usd",1)
    assert ml.budget_stop_reason(b) is not None
    assert "不代表免費" in ml.budget_report(b)


def test_budget_corruption_does_not_reset_spend(monkeypatch,tmp_path):
    import mining_loop as ml
    p=tmp_path/"budget.json";p.write_text("broken")
    monkeypatch.setattr(ml,"BUDGET_PATH",p)
    with pytest.raises(RuntimeError):ml.load_budget()
    assert p.read_text()=="broken"


def test_pending_usage_stops_next_call(monkeypatch,tmp_path):
    import mining_loop as ml
    monkeypatch.setattr(ml,"BUDGET_PATH",tmp_path/"budget.json")
    monkeypatch.setitem(ml.BUD,"weekly_token_budget",10)
    monkeypatch.setitem(ml.BUD,"weekly_cost_budget_usd",0)
    m=ml.Meter();m.add(ml.LLMResult("",tokens=10,cost_known=True))
    def forbidden(*a,**k):raise AssertionError("must not launch another process")
    monkeypatch.setattr(ml.subprocess,"run",forbidden)
    with pytest.raises(RuntimeError,match="token"):ml.call_llm("prompt",meter=m)
    assert m.calls==1
