"""Scope decisions, source validation, no-lookahead, journal/rollback and idempotency."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import factor_scope as fs
from memory import Memory


@pytest.fixture
def project(tmp_path):
    cfg = copy.deepcopy(fs.ec.CFG)
    cfg['paths']['monthly_base'] = 'data/monthly_base.parquet'
    cfg['paths']['memory_dir'] = 'memory'
    cfg['scope_control'] = {'classification_source': 'data/universe.parquet',
        'groups': {'Old': [], 'New': [], 'Renamed': ['Old']},
        'legacy_factor_ids': ['F-001'], 'legacy_groups': ['Old'], 'max_monthly_turnover': 0.4}
    (tmp_path/'data').mkdir(); (tmp_path/'memory').mkdir()
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    (tmp_path/'fields.yaml').write_text('test: {x: {}}',encoding='utf-8')
    rng=np.random.default_rng(72)
    months=pd.period_range('2012-01',periods=110,freq='M').astype(str)
    records=[]
    for month in months:
        for g in ['Old','New']:
            for i in range(20):
                records.append((g+str(i),month,g,float(i),i*.02+rng.normal(0,.05)))
    base=pd.DataFrame(records,columns=['stock_id','ym','group','x','fwd_ret_1m'])
    base.to_parquet(tmp_path/'data/monthly_base.parquet',index=False)
    base[['stock_id','group']].drop_duplicates().to_parquet(tmp_path/'data/universe.parquet',index=False)
    meta={'name_zh':'Test','formula':'cs_rank(x)','industry_scope':'Old',
          'category':'test','test_metrics_sealed':{'secret_test':12345},'sub_train':{'icir':1.1}}
    fs.write_json(tmp_path/'memory/library.json',{'F-001':meta,'R-001':{'reference':True}})
    pd.DataFrame([('F-001','2012-01','Old0',.1),('R-001','2012-01','Old0',.5)],
        columns=['factor_id','ym','stock_id','value']).to_parquet(tmp_path/'memory/factor_values.parquet',index=False)
    return tmp_path,base,cfg


def test_classification_conflicts_unknown_and_lineage(project):
    root,b,cfg=project
    u=b[['stock_id','group']].drop_duplicates()
    p=cfg['scope_control']
    assert fs.validate_panel(b,u,p)['classification_conflicts']==0
    bad=b.copy();bad.loc[0,'group']='Unknown'
    with pytest.raises(ValueError,match='未知'): fs.validate_panel(bad,u,p)
    with pytest.raises(ValueError,match='重複'): fs.validate_panel(pd.concat([b,b.iloc[:1]]),u,p)
    badu=pd.concat([u,pd.DataFrame([{'stock_id':'Old0','group':'New'}])])
    with pytest.raises(ValueError,match='衝突'): fs.validate_panel(b,badu,p)
    with pytest.raises(ValueError,match='循環'): fs.ancestors('A',{'A':['B'],'B':['A']})
    m={'groups_seen':['Old'],'excluded_at_admission':['Old']}
    assert fs.candidate_groups(m,['Old','New','Renamed'],p['groups'])==['New']


def test_apply_preserves_history_and_reference_and_is_idempotent(project):
    root,b,cfg=project
    original=Memory(root/'memory').library()
    result=fs.prepare(root,progress=lambda _:None)
    assert result['status']=='ready',result
    item=result['factors'][0]
    assert item['before']==['Old'] and item['after']==['New','Old']
    assert Memory(root/'memory').library()==original  # check-only is read-only for active data
    assert 'secret_test' not in json.dumps(result)
    fs.apply(result['run_id'],root)
    lib,values=Memory(root/'memory').snapshot()
    assert lib['F-001']['test_metrics_sealed']==original['F-001']['test_metrics_sealed']
    assert lib['F-001']['sub_train']==original['F-001']['sub_train']
    assert lib['F-001']['groups_seen']==['New','Old']
    assert values[values.factor_id=='R-001'].value.tolist()==[.5]
    assert fs.prepare(root,progress=lambda _:None)['status']=='no_change'
    with pytest.raises(ValueError,match='ready'):fs.apply(result['run_id'],root)


def test_test_period_poison_does_not_change_decisions(project):
    root,b,cfg=project
    first=fs.prepare(root,progress=lambda _:None)
    b.loc[b.ym>='2020-01',['x','fwd_ret_1m']]=-9999
    b.to_parquet(root/'data/monthly_base.parquet',index=False)
    second=fs.prepare(root,progress=lambda _:None)
    assert first['status']==second['status']=='ready'
    assert first['factors'][0]['assessments']==second['factors'][0]['assessments']


def test_failed_new_group_is_seen_old_exclusions_never_readded(project):
    root,b,cfg=project
    b.loc[b.group=='New','fwd_ret_1m']*=-1
    b.to_parquet(root/'data/monthly_base.parquet',index=False)
    result=fs.prepare(root,progress=lambda _:None)
    assert result['factors'][0]['after']==['Old']
    assert result['factors'][0]['assessments'][0]['status']=='rejected'
    fs.apply(result['run_id'],root)
    lib=Memory(root/'memory').library()
    assert 'New' in lib['F-001']['groups_seen']
    assert fs.candidate_groups(lib['F-001'],['Old','New'],cfg['scope_control']['groups'])==[]


def test_insufficient_data_waits_without_marking_seen(project):
    root,b,cfg=project
    b.loc[(b.group=='New') & (b.ym<'2017-01'),'fwd_ret_1m']=np.nan
    b.to_parquet(root/'data/monthly_base.parquet',index=False)
    result=fs.prepare(root,progress=lambda _:None)
    assert result['factors'][0]['assessments'][0]['status']=='insufficient_data'
    fs.apply(result['run_id'],root)
    assert 'New' not in Memory(root/'memory').library()['F-001']['groups_seen']


def test_stale_plan_and_tampered_staging_rejected(project):
    root,b,cfg=project
    result=fs.prepare(root,progress=lambda _:None)
    old=fs.digest(root/'memory/library.json')
    (root/'fields.yaml').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='改變'):fs.apply(result['run_id'],root)
    assert fs.digest(root/'memory/library.json')==old


def test_commit_failure_rolls_back_both_files(project,monkeypatch):
    root,b,cfg=project
    result=fs.prepare(root,progress=lambda _:None)
    before={n:fs.digest(root/'memory'/n) for n in ['library.json','factor_values.parquet']}
    replace=fs.os.replace
    def fail_library(src,dest):
        if Path(src).name=='library.ready':raise OSError('simulated interrupted commit')
        return replace(src,dest)
    monkeypatch.setattr(fs.os,'replace',fail_library)
    with pytest.raises(OSError):fs.apply(result['run_id'],root)
    assert before=={n:fs.digest(root/'memory'/n) for n in before}
    assert not (root/'memory/.scope_pending.json').exists()
    assert not (root/'memory/.library.lock').exists()


def test_scope_lock_blocks_admission_and_read(project):
    root,_,_=project
    a=Memory(root/'memory');b=Memory(root/'memory')
    with a.locked():
        with pytest.raises(RuntimeError,match='使用'):b.library()


def test_unknown_provenance_is_blocked(project):
    root,_,_=project
    lib=Memory(root/'memory').library();lib['F-099']=lib.pop('F-001')
    fs.write_json(root/'memory/library.json',lib)
    result=fs.prepare(root,progress=lambda _:None)
    assert result['status']=='blocked' and '來源' in result['error']


def test_interrupted_commit_recovery(project):
    import shutil
    root,_,_=project
    result=fs.prepare(root,progress=lambda _:None)
    mem=Memory(root/'memory');folder=fs.run_folder(mem,result['run_id'])
    for key,path in [('library',mem.lib_path),('values',mem.values_path)]:
        shutil.copy2(path,folder/(key+'.before'))
    fs.write_json(mem.root/'.scope_pending.json',{'run_id':result['run_id']})
    mem.lib_path.write_text('{}',encoding='utf-8')
    with pytest.raises(RuntimeError,match='未完成'):mem.library()
    restored=fs.recover(root)
    assert restored['status']=='rolled_back'
    assert fs.digest(mem.lib_path)==result['inputs']['library']
    assert fs.digest(mem.values_path)==result['inputs']['values']


def test_old_excluded_group_is_never_evaluated_even_if_current(project):
    root,_,cfg=project
    p=cfg['scope_control']
    p['legacy_groups']=['Old','New']
    meta=fs.bootstrap('F-001',{'industry_scope':'Old'},p)
    assert meta['excluded_at_admission']==['New']
    assert fs.candidate_groups(meta,['Old','New','Renamed'],p['groups'])==[]


def test_short_only_requires_same_use(monkeypatch,project):
    root,base,cfg=project
    ctx=fs.context_for(base,cfg)
    fac=fs.dsl.compute('cs_rank(x)',ctx.data,ctx.group_map)
    monkeypatch.setattr(fs.ec,'industry_qualify',lambda *a:[('New',1.,1.,0.)])
    monkeypatch.setattr(fs.ec,'leg_stats',lambda *a,**k:({'long_excess_ann':.1,'short_excess_ann':-.1,'short_turnover_m':0.},0.))
    assert fs.assess_group(ctx,fac,'New','short_only',cfg['scope_control'])['status']=='rejected'
    assert fs.assess_group(ctx,fac,'New','long_only',cfg['scope_control'])['status']=='passed'
