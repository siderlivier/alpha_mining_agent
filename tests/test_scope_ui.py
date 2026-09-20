"""HTTP contract and command boundary; no real mining or production writes."""
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest
import scope_ui


class FakeApp:
    token='test-token'
    calls=[]
    def state(self):
        return {'reports':[],'job':None}
    def start(self,action,run_id=None):
        if action=='busy':raise RuntimeError('busy')
        if action!='check':raise ValueError('bad action')
        self.calls.append(action)
        return {'job_id':'test-job'}


@pytest.fixture
def server():
    app=FakeApp();app.calls=[]
    http=scope_ui.make_server(0,app)
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    yield http,app
    http.shutdown();http.server_close();thread.join()


def request(server,path='/',payload=None,token=None,origin=None,host=None):
    http,_=server
    headers={}
    if token:headers['X-CSRF-Token']=token
    if origin:headers['Origin']=origin
    if host:headers['Host']=host
    data=None if payload is None else json.dumps(payload).encode()
    req=Request(f'http://127.0.0.1:{http.server_port}{path}',data=data,headers=headers)
    try:
        with urlopen(req,timeout=3) as response:return response.status,response.read().decode(),response.headers
    except HTTPError as exc:return exc.code,exc.read().decode(),exc.headers


def test_page_and_readonly_state(server):
    code,text,headers=request(server)
    assert code==200 and 'test-token' in text and '開始檢查' in text
    assert 'frame-ancestors' in headers['Content-Security-Policy']
    assert request(server,'/api/state')[0]==200
    assert not server[1].calls


def test_mutations_require_token_and_same_origin(server):
    assert request(server,'/api/jobs',{'action':'check'})[0]==403
    assert request(server,'/api/jobs',{'action':'check'},'test-token','https://evil.example')[0]==403
    assert request(server,'/api/jobs',{'action':'check'},'test-token')[0]==202
    assert server[1].calls==['check']


def test_unknown_paths_and_host_rejected(server):
    assert request(server,'/../../memory/library.json')[0]==404
    assert request(server,host='evil.example')[0]==403
    assert request(server,'/api/jobs',{'action':'shell'},'test-token')[0]==400
    assert request(server,'/api/jobs',{'action':'busy'},'test-token')[0]==409


def test_command_allowlist_and_duplicate_job_guard(tmp_path):
    app=object.__new__(scope_ui.Application)
    app.mutex=threading.Lock();app.job={'status':'running'}
    with pytest.raises(ValueError):app.start('shell')
    with pytest.raises(ValueError):app.start('apply','../../bad')
    with pytest.raises(RuntimeError):app.start('check')
