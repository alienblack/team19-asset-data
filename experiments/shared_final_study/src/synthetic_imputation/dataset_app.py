"""Loopback-only browser workflow, using Python's standard-library HTTP server."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import threading
from urllib.parse import unquote, urlsplit
import uuid
import webbrowser

from jinja2 import Environment, FileSystemLoader, select_autoescape
from .dataset import inspect_dataset, json_ready, run_dataset
from .dataset_models import METHODS


def create_server(project_root, port=8765):
    root = Path(project_root).resolve()
    storage = root/'outputs/dataset-workbench'
    token = secrets.token_urlsafe(24)
    uploads,jobs = {},{}
    lock = threading.Lock()
    templates = Path(__file__).parent/'templates'
    environment = Environment(loader=FileSystemLoader(templates),autoescape=select_autoescape(['html']))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message, *args):
            return

        def send_json(self, value, status=200):
            data = json.dumps(json_ready(value),allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

        def send_file(self, base, relative):
            base = base.resolve()
            path = (base/unquote(relative)).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                self.send_error(404);return
            data = path.read_bytes()
            self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            if path.suffix in ('.csv','.json'):self.send_header('Content-Disposition','attachment; filename="'+path.name.replace('"','')+'"')
            self.end_headers();self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path=='/':
                page = environment.get_template('workbench.html').render(token=token,methods=METHODS)
                data=page.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            elif path.startswith('/api/jobs/'):
                with lock: job = dict(jobs.get(path.split('/')[-1],{}))
                self.send_json(job,200 if job else 404)
            elif path=='/api/example':
                try:
                    source=root/'examples/example_assets.csv'
                    identifier=uuid.uuid4().hex
                    uploads[identifier]=(source,None)
                    profile=inspect_dataset(source)
                    settings=json.loads((root/'examples/example_config.json').read_text())
                    overrides={s['name']:s for s in settings['columns']}
                    for column in profile['columns']:column.update(overrides[column['name']])
                    self.send_json({'id':identifier,'profile':profile})
                except Exception as error:self.send_json({'error':str(error)},400)
            elif path.startswith('/files/'):
                parts=path.split('/',3)
                with lock:job=jobs.get(parts[2])
                if len(parts)==4 and job:self.send_file(Path(job['output']),parts[3])
                else:self.send_error(404)
            elif path.startswith('/benchmark/'):
                relative=path[len('/benchmark/'):]
                if relative.startswith('reports/published/'):
                    self.send_file(root/'reports/published',relative[len('reports/published/'):])
                elif relative.startswith('reference_results/'):
                    self.send_file(root/'reference_results',relative[len('reference_results/'):])
                else:self.send_error(404)
            else:self.send_error(404)

        def do_POST(self):
            if self.headers.get('X-Team19-Token')!=token:
                self.send_json({'error':'Reload the workbench before submitting a request'},403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=140_000_000:raise ValueError('Request too large; use CSVs below 50 MB each')
                value=json.loads(self.rfile.read(size))
                if self.path=='/api/inspect':
                    identifier=uuid.uuid4().hex
                    folder=storage/'uploads'/identifier;folder.mkdir(parents=True)
                    def uploaded(key,name_key):
                        if not value.get(key):return None
                        name=Path(value.get(name_key,'dataset.csv')).name
                        if not name.lower().endswith('.csv'):raise ValueError('Choose a CSV file')
                        data=base64.b64decode(value[key],validate=True)
                        if len(data)>50_000_000:raise ValueError('CSV exceeds 50 MB')
                        dest=folder/(('truth-' if key=='truth_content' else '')+name)
                        dest.write_bytes(data);return dest
                    source=uploaded('content','name');truth=uploaded('truth_content','truth_name')
                    if source is None:raise ValueError('Choose a dataset first')
                    profile=inspect_dataset(source)
                    if truth is not None:inspect_dataset(truth)
                    uploads[identifier]=(source,truth)
                    self.send_json({'id':identifier,'profile':profile,'truth_supplied':truth is not None})
                elif self.path=='/api/run':
                    if value.get('id') not in uploads:raise ValueError('Choose and inspect a dataset first')
                    with lock:
                        if any(j['status']=='running' for j in jobs.values()):
                            self.send_json({'error':'A study is already running. Wait for it to finish.'},409);return
                        identifier=uuid.uuid4().hex[:12]
                        out=storage/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+identifier)
                        jobs[identifier]={'id':identifier,'status':'running','progress':0,
                                          'message':'Checking dataset and settings','events':[], 'output':str(out)}
                    source,truth=uploads[value['id']]
                    config=value.get('config',{})
                    def update(message,done,total):
                        with lock:
                            jobs[identifier].update(message=message,progress=round(100*done/max(total,1)))
                            jobs[identifier]['events']=(jobs[identifier]['events']+[message])[-50:]
                    def execute():
                        try:
                            report=run_dataset(source,out,config,truth,progress=update)
                            status=json.loads((out/'run_status.json').read_text())
                            with lock:jobs[identifier].update(status=status['status'],progress=100,
                                report=f'/files/{identifier}/{report.name}',message='Study complete')
                        except Exception as error:
                            with lock:jobs[identifier].update(status='failed',message=str(error),error=str(error))
                    threading.Thread(target=execute,daemon=True).start()
                    self.send_json({'job':identifier})
                else:self.send_json({'error':'Unknown endpoint'},404)
            except Exception as error:self.send_json({'error':str(error)},400)

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server


def serve(project_root,port=8765,open_browser=True):
    server=create_server(project_root,port)
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'Team 19 dataset workbench: {url}\nFiles and processing stay on this computer. Ctrl+C stops the workbench.',flush=True)
    if open_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
    return 0
