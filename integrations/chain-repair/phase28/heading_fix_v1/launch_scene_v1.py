"""Start exactly one pinned GPU run and postprocess; never restore old snapshots."""

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

import datetime
import fcntl
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]


def main():
    os.chdir(ROOT)
    manifest=_hugsim_expand(json.loads((HERE/'scene_launch_manifest.json').read_text()))
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Pinned input/code changed: '+name)
    req=_hugsim_expand(json.loads((HERE/'scene552_request_v1.json').read_text()))
    output=Path(req['output'])
    if output.exists():
        raise RuntimeError('Output already exists; refusing duplicate/overwrite: '+str(output))
    if (ROOT/'outputs/gpu_test_service/STOP').exists():
        raise RuntimeError('GPU service STOP exists; inspect reason first')
    lock=(ROOT/'outputs/gpu_test_service/worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name)==os.getpid():continue
        try:
            executable=(directory/'comm').read_text()
            command=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if ('python' in executable or 'pt_main' in executable) and any(x in command for x in ['patch_run.py','heading_scene_run_v1.py','branch_worker.py','run_arm.py','track_run.py']):
            raise RuntimeError('Existing simulation process: '+directory.name)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this terminal; run this command in the GPU terminal')
    smi=shutil.which('nvidia-smi')
    if smi:
        rows=subprocess.check_output([smi,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        active=[r.strip() for r in rows.splitlines() if r.strip().isdigit() and int(r.strip())!=os.getpid()]
        if active:raise RuntimeError('GPU already has compute processes: '+','.join(active))
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    print('Starting one fresh 552 run; output:',output,flush=True)
    with (HERE/'scene552_v1.log').open('x') as log:
        log.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');log.flush()
        code=subprocess.call([sys.executable,str(P28/'heading_scene_run_v1.py'),str(HERE/'scene552_request_v1.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
        if (output/'run_report.json').exists():
            post=subprocess.call([sys.executable,str(HERE/'postprocess_scene_v1.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
        else:post=1
    print('Simulation exit:',code,'postprocessing exit:',post,'log:',HERE/'scene552_v1.log',flush=True)
    if code or post:raise SystemExit(1)


if __name__=='__main__':main()
