
# Portable source/data boundary (no global pathlib patch).
import sys as _hugsim_sys
from pathlib import Path as _HugsimNativePath
_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents if (p / "integrations/_shared/paths.py").is_file()), None)
if _hugsim_repository is None:
    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")
if str(_hugsim_repository) not in _hugsim_sys.path:
    _hugsim_sys.path.insert(0, str(_hugsim_repository))
from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand
Path = _HugsimPath

import argparse,json
from platform_io import read,DEFAULT_STORE
from platform_catalog import Catalog,register
from platform_runner import Runner
from platform_service import simulate,stdio
from platform_contracts import tool_specs

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['init','capabilities','tools','prepare','check','run','sensor-validate','simulate-llm','stdio','status'])
    p.add_argument('input',nargs='?');p.add_argument('--store',default=str(DEFAULT_STORE));p.add_argument('--request-id',default=None)
    p.add_argument('--output');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.action=='init':register();result=dict(registered=True,catalog=str(Catalog().path))
    else:
        runner=Runner(a.store)
        if a.action=='stdio':stdio(runner);return
        if a.action=='capabilities':result=runner.catalog.capabilities()
        elif a.action=='tools':result=tool_specs()
        elif a.action=='status':
            pass  # Path is provided by the portable boundary above
            from platform_io import identity
            result=read(Path(a.store)/'jobs'/identity(a.request_id)/'state.json')
        elif a.action=='simulate-llm':
            if not a.input or not a.output:p.error('simulate-llm requires goal JSON and --output')
            result=simulate(read(a.input),runner,a.output,a.prepare_only)
        else:
            if not a.input or not a.request_id:p.error('task path and --request-id are required')
            task=read(a.input)
            if a.action=='prepare':result=dict(job=str(runner.prepare(task,a.request_id)),controls_executed=False)
            else:result=getattr(runner,a.action.replace('-','_'))(task,a.request_id)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
