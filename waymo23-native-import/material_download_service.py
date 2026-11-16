"""Single-worker, allowlisted material download queue; no shell commands."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE/'outputs/material_download_service'


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def command(request):
    if set(request) != {'id', 'action'}:
        raise ValueError('Only id and action are accepted')
    actions = {
        'list_source': ['probe_3drealcar_download.py'],
        'probe_download': ['download_3drealcar_vehicle.py', '--max-archives', '1'],
        'download_vehicle': ['download_3drealcar_vehicle.py'],
    }
    args = actions[request['action']]
    return [sys.executable, str(BASE/'waymo23-native-import'/args[0]), *args[1:]]


def serve():
    (ROOT/'requests').mkdir(parents=True, exist_ok=True)
    (ROOT/'jobs').mkdir(exist_ok=True)
    with (ROOT/'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_work = time.monotonic()
        completed = 0
        heartbeat = dict(pid=os.getpid(), status='idle')
        try:
            print('Material download executor ready:', ROOT, flush=True)
            while completed < 10 and time.monotonic()-last_work < 1800:
                if (ROOT/'STOP').exists():
                    break
                heartbeat.update(status='idle', completed_jobs=completed, updated_epoch=time.time())
                atomic(ROOT/'heartbeat.json', heartbeat)
                pending = sorted((ROOT/'requests').glob('*.json'))
                if not pending:
                    time.sleep(2)
                    continue
                queued = pending[0]
                request = json.loads(queued.read_text())
                if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', request.get('id', '')) or request['id'] != queued.stem:
                    raise ValueError('Invalid job identity')
                argv = command(request)
                folder = ROOT/'jobs'/request['id']
                folder.mkdir(exist_ok=False)
                queued.replace(folder/'request.json')
                status = dict(id=request['id'], status='running', started_epoch=time.time())
                atomic(folder/'status.json', status)
                process = None
                try:
                    with (folder/'run.log').open('w') as log:
                        process = subprocess.Popen(argv, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                                                   env=dict(os.environ, PYTHONUNBUFFERED='1'), start_new_session=True)
                        began = time.monotonic()
                        while process.poll() is None:
                            heartbeat.update(status='running', job=request['id'], updated_epoch=time.time())
                            atomic(ROOT/'heartbeat.json', heartbeat)
                            if time.monotonic()-began > 2100 or (ROOT/'STOP').exists():
                                raise RuntimeError('Download timeout or STOP requested')
                            time.sleep(2)
                    status.update(status='passed' if process.returncode == 0 else 'failed', exit_code=process.returncode)
                except BaseException as exc:
                    status.update(status='interrupted', error=str(exc))
                    raise
                finally:
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    status['finished_epoch'] = time.time()
                    atomic(folder/'status.json', status)
                print(request['id'], status['status'], flush=True)
                completed += 1
                last_work = time.monotonic()
        finally:
            heartbeat.update(status='stopped', updated_epoch=time.time())
            atomic(ROOT/'heartbeat.json', heartbeat)


if __name__ == '__main__':
    serve()
