"""Local factor-scope UI: python -B src/scope_ui.py --port 8765.

Loopback only; fixed commands, CSRF protection, one child job at a time.
No extra web framework dependency. Do not expose this service to the internet.
"""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import yaml

ROOT = Path(__file__).resolve().parents[1]


class Application:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.token = secrets.token_urlsafe(32)
        cfg = self.config()
        self.memory = self.root / cfg['paths']['memory_dir']
        self.jobs_dir = self.memory / 'scope_jobs'
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.job = None
        self.mutex = threading.Lock()

    def config(self):
        return yaml.safe_load((self.root / 'config.yaml').read_text(encoding='utf-8'))

    def reports(self):
        reports = []
        for path in sorted((self.memory / 'scope_runs').glob('*/report.json'), reverse=True)[:20]:
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                reports.append(report)
            except (OSError, json.JSONDecodeError):
                continue
        return reports

    def state(self):
        cfg = self.config()
        with self.mutex:
            job = dict(self.job) if self.job else None
        if job:
            path = self.jobs_dir / (job['id'] + '.log')
            if path.exists():
                with path.open('rb') as handle:
                    handle.seek(max(0, path.stat().st_size - 16000))
                    job['log'] = handle.read().decode('utf-8', errors='replace')
        reports = self.reports()
        if any(r.get('status') == 'ready' for r in reports):
            import factor_scope
            current = {k: factor_scope.digest(p) for k, p in factor_scope.input_paths(self.root, cfg).items()}
            for r in reports:
                r['stale'] = r.get('status') == 'ready' and r.get('inputs') != current
        return {'job': job, 'reports': reports,
                'thresholds': cfg['funnel']['stage4b_industry'],
                'turnover_limit': cfg['scope_control']['max_monthly_turnover'],
                'groups': cfg['scope_control']['groups'],
                'pending_recovery': (self.memory / '.scope_pending.json').exists(),
                'library_locked': (self.memory / '.library.lock').exists()}

    def start(self, action, run_id=None):
        if action not in ('check', 'auto_apply', 'apply', 'recover'):
            raise ValueError('未知操作')
        if action == 'apply':
            import re
            if not isinstance(run_id, str) or not re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{8}', run_id):
                raise ValueError('不合法 run_id')
        with self.mutex:
            if self.job and self.job['status'] == 'running':
                raise RuntimeError('已有工作執行中')
            job_id = uuid.uuid4().hex
            self.job = {'id': job_id, 'status': 'running', 'action': action}
        args = {'check': ['--check-only'], 'auto_apply': ['--auto-apply'],
                'apply': ['--apply', run_id], 'recover': ['--recover']}[action]
        thread = threading.Thread(target=self._work, args=(job_id, args), daemon=True)
        thread.start()
        return {'job_id': job_id}

    def _work(self, job_id, args):
        result_path = self.jobs_dir / (job_id + '.json')
        try:
            with (self.jobs_dir / (job_id + '.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-u', '-B',
                    str(self.root / 'src/factor_scope.py'), *args,
                    '--result-file', str(result_path)], cwd=self.root,
                    stdout=log, stderr=subprocess.STDOUT, shell=False,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            payload = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
            status = 'completed' if result.returncode == 0 else 'failed'
            with self.mutex:
                self.job.update(status=status, result=payload, returncode=result.returncode)
        except Exception as exc:
            with self.mutex:
                self.job.update(status='failed', error=str(exc))


def make_server(port=8765, app=None):
    app = app or Application()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _allowed_host(self):
            return self.headers.get('Host') in (
                f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def reply(self, code, payload, content_type='application/json; charset=utf-8'):
            data = payload.encode('utf-8') if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self._allowed_host():
                return self.reply(403, {'error': '禁止遠端 Host'})
            if self.path == '/':
                page = Path(__file__).with_name('scope_ui.html').read_text(encoding='utf-8')
                return self.reply(200, page.replace('__CSRF_TOKEN__', app.token), 'text/html; charset=utf-8')
            if self.path == '/api/state':
                try:
                    return self.reply(200, app.state())
                except Exception as exc:
                    return self.reply(500, {'error': str(exc)})
            return self.reply(404, {'error': '找不到頁面'})

        def do_POST(self):
            if not self._allowed_host():
                return self.reply(403, {'error': '禁止遠端 Host'})
            origin = self.headers.get('Origin')
            allowed_origins = (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}')
            if (origin and origin not in allowed_origins) or not secrets.compare_digest(
                    self.headers.get('X-CSRF-Token', ''), app.token):
                return self.reply(403, {'error': '來源或操作 token 不符'})
            if self.path != '/api/jobs':
                return self.reply(404, {'error': '未知操作'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096:
                    raise ValueError('請求大小不合法')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('請求必須是物件')
                result = app.start(body.get('action'), body.get('run_id'))
                return self.reply(202, result)
            except RuntimeError as exc:
                return self.reply(409, {'error': str(exc)})
            except (ValueError, TypeError) as exc:
                return self.reply(400, {'error': str(exc)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    server = make_server(args.port)
    print(f'Factor scope UI: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
