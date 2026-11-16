"""Dependency-free persistent JSONL client usable by a future LLM tool executor."""

# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand

import json,subprocess

class Client:
    def __init__(self,command):
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def call(self,request_id,tool,arguments):
        self.process.stdin.write(json.dumps(dict(request_id=request_id,tool=tool,arguments=arguments),allow_nan=False)+'\n')
        self.process.stdin.flush();line=self.process.stdout.readline()
        if not line:raise RuntimeError('Platform process exited before returning a response')
        response=_hugsim_expand(json.loads(line))
        if response.get('request_id') not in (None,request_id):raise RuntimeError('Response request identity mismatch')
        return response
    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
            self.process.stdout.close()
        except subprocess.TimeoutExpired:
            raise RuntimeError('Service is still executing; owner left running to retain GPU lock and cleanup. Inspect durable job state before recovery.')
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
