"""Durable background asset preparation and patch-off/on discovery probes."""

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

import argparse
import fcntl
import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import signal
import subprocess
import sys
import time
import traceback

LOCAL=Path(__file__).resolve().parent
ROOT=LOCAL.parents[1]
P26=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair/phase26'
OUT=P26/'patch_task_discovery_v1'
AGENT=ROOT/'integrations/autonomous-patch-agent-v7'
ASSETS=ROOT/'integrations/patch-generalization-assets-v1'
ASSET_OUT=P26/'patch_generalization_assets_v1'
CAMPAIGN=P26/'autonomous_patch_agent_v7/campaigns/patch_auto_v7'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return _hugsim_expand(json.loads(Path(path).read_text()))


def write(path,value):
    import tempfile
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def asset(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def verify(receipt):
    if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Changed frozen source: '+receipt['path'])


def identity(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text();tail=text[text.rfind(')')+2:].split()
        if tail[0]=='Z':return None
        return (Path('/proc/sys/kernel/random/boot_id').read_text().strip(),tail[19])
    except (OSError,IndexError):return None


def alive(owner):return bool(owner and identity(owner.get('pid')) and list(identity(owner['pid']))==owner.get('identity'))


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'contract.json'
    if path.exists():
        c=read(path)
        for p,h in c['source_pins'].items():verify(dict(path=p,sha256=h))
        for arm in c['policies'].values():
            for p,h in arm['code_pins'].items():verify(dict(path=p,sha256=h))
        return c
    state=read(CAMPAIGN/'state.json');registry=read(CAMPAIGN/'candidate_registry.json')
    best=max((x for x in registry.values() if x.get('metrics',{}).get('cutin551',{}).get('status')=='completed'),key=lambda x:x['metrics']['cutin551']['score']['hdscore'])
    c=dict(schema='background_scene_preparation_v1',campaign_snapshot=asset(CAMPAIGN/'state.json'),registry_snapshot=asset(CAMPAIGN/'candidate_registry.json'),policies=dict(primary=best,stable=state['champion']),
        source_pins={str(p):sha(p) for p in LOCAL.rglob('*') if p.suffix in ('.py','.sh') and '__pycache__' not in p.parts},
        budgets=dict(asset_wall_seconds=86400,pair_wall_seconds=14400,pair_max_episodes=4),
        budget_scope='New user-requested asset preparation and diagnostic off/on comparison; existing patch campaign budgets/state unchanged. These are not new LLM patch rounds.',
        definition='Find improvement by comparing native UniAD + shared iLQR with full external patch under identical edited world; no assumed outcome',
        task=dict(id='scene551_cutin_off_on_probe',scene=551,program='registered_native_cutin_duration_variants',parameters={}),
        control_policy_changed=False,automatic_patch_task_admitted=False,global_patch_validated=False)
    write(path,c);return c


def start(kind):
    freeze();folder=OUT/kind;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'start.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner=read(folder/'owner.json') if (folder/'owner.json').exists() else {}
        if alive(owner):return dict(status='already_running',kind=kind,pid=owner['pid'])
        if (folder/'STOP').exists():return dict(status='stopped',kind=kind,note='use resume-assets or resume-pair')
        env={k:v for k,v in os.environ.items() if k not in ('PATCH_API_KEY','PATCH_API_BASE_URL','PATCH_MODEL','OPENAI_API_KEY')}
        mark(kind,'starting',episodes_completed=0 if kind=='pair' else None)
        with (folder/'worker.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u','-B',str(LOCAL/'service.py'),'_worker','--kind',kind],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        write(folder/'owner.json',dict(pid=proc.pid,identity=identity(proc.pid)))
        return dict(status='started',kind=kind,pid=proc.pid,log=str(folder/'worker.log'))


def status():
    result={}
    for kind in ('assets','pair'):
        folder=OUT/kind
        s=read(folder/'status.json') if (folder/'status.json').exists() else {'stage':'not_started'}
        owner=read(folder/'owner.json') if (folder/'owner.json').exists() else {}
        result[kind]=dict(s,alive=alive(owner))
    return result


def stopped(kind):return (OUT/kind/'STOP').exists() or (ROOT/'outputs/gpu_test_service/STOP').exists()


def gpu_available():
    import torch
    return torch.cuda.is_available()


def gpu_lock_free():
    with (ROOT/'outputs/gpu_test_service/worker.lock').open('r') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        fcntl.flock(lock,fcntl.LOCK_UN);return True


def mark(kind,stage,**extra):
    write(OUT/kind/'status.json',dict(stage=stage,updated_epoch=time.time(),**extra))


def gpu_busy_receipt(receipt):
    # Both the reserved exit code and the exact host-generated message are required.
    if receipt.get('returncode') not in (1,75) or not receipt.get('log'):return False
    verify(receipt['log'])
    return Path(receipt['log']['path']).read_text().strip()=='GPU lock busy'


def count_completed(kind):
    return sum(1 for f in (OUT/kind/'episodes').glob('*/completion.json'))


def command(kind,tag,args,wall_limit):
    base=OUT/kind/'jobs'/tag
    retry=0
    while True:
        folder=base if retry==0 else base.with_name(base.name+'_retry'+str(retry).zfill(3))
        folder.mkdir(parents=True,exist_ok=True)
        done=folder/'receipt.json'
        if not done.exists():break
        receipt=read(done);verify(receipt['log'])
        if receipt['status']=='succeeded':return True
        if not gpu_busy_receipt(receipt):
            raise RuntimeError('Previous job incomplete/failed, artifacts preserved: '+str(folder))
        retry+=1
    ownerfile=folder/'owner.json'
    if ownerfile.exists():
        owner=read(ownerfile)
        if alive(owner):
            mark(kind,'observing_existing_job',job=tag)
            while alive(owner) and not stopped(kind):time.sleep(2)
            if done.exists():return read(done)['status']=='succeeded'
        raise RuntimeError('Interrupted job retained for diagnosis: '+str(folder))
    logpath=folder/'output.log'
    if logpath.exists():raise RuntimeError('Ambiguous partial dispatch retained: '+str(folder))
    mark(kind,'running',job=tag)
    with logpath.open('x') as log:
        p=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        write(ownerfile,dict(pid=p.pid,identity=identity(p.pid),command=args))
        began=time.monotonic()
        while p.poll() is None:
            if stopped(kind) or time.monotonic()-began>wall_limit:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(1)
    receipt=dict(status='succeeded' if p.returncode==0 else 'failed',returncode=p.returncode,log=asset(logpath),seconds=time.monotonic()-began)
    write(done,receipt)
    if gpu_busy_receipt(receipt):
        mark(kind,'waiting_gpu',job=tag,gpu_busy_retries=retry+1,episodes_completed=count_completed(kind))
        return False
    if p.returncode:raise RuntimeError('Job failed; inspect '+str(logpath))
    return True


def assets_tick(contract):
    # CPU work runs immediately; GPU semantic work waits without monopolizing GPU.
    for scene in read(ASSETS/'cases.json'):
        for stage in ('prepare','metadata'):
            if stopped('assets'):return False
            command('assets',scene+'_'+stage,['bash',str(ASSETS/'run.sh'),stage,'--scene',scene],7200)
        bundle=ASSET_OUT/('scene'+scene)/'bundle'
        if not (bundle/'map_support/manifest.json').exists():
            from road_assets import export
            export(bundle)
    if not gpu_available() or not gpu_lock_free():
        mark('assets','waiting_gpu',cpu_preparation_complete=True,remaining=['semantic19','point_semantics','scene runtime/map admission'])
        return False
    for scene in read(ASSETS/'cases.json'):
        # A requested off/on pair has priority over bulk segmentation.
        pair=read(OUT/'pair/status.json') if (OUT/'pair/status.json').exists() else {}
        if pair.get('stage') not in (None,'completed','error','stopped'):
            mark('assets','waiting_pair_priority');return False
        for stage in ('semantics','points'):
            if stopped('assets'):return False
            if not command('assets',scene+'_'+stage,['bash',str(ASSETS/'run.sh'),stage,'--scene',scene],21600):return False
    mark('assets','source_assets_exported_admission_pending',remaining=['validated per-scene runtime/map/native-score integration'],global_patch_validated=False)
    return True


def pair_tick(contract):
    if not gpu_available() or not gpu_lock_free():mark('pair','waiting_gpu',episodes_completed=count_completed('pair'));return False
    from episode import prepare
    task=contract['task'];champ=contract['policies']['primary'];root=OUT/'pair';work=root/'episodes'
    cpu=root/'preflight.json'
    command('pair','preflight',[sys.executable,'-u','-B',str(LOCAL/'service.py'),'_preflight'],1800)
    # Both original and edited scenes get fresh native and patched histories.
    count=0
    for mode in ('off','on'):
        for edited in (False,True):
            tag=('edited' if edited else 'original')+'_'+mode;folder=work/tag
            if not folder.exists():prepare(task,champ,folder)
            if not (folder/'comparison_mode.json').exists():write(folder/'comparison_mode.json',dict(mode=mode,edited=edited,task=task))
            # A complete immutable receipt is the only reuse allowed; no action replay.
            if not command('pair',tag,['bash',str(LOCAL/'run.sh'),'_episode','--episode',tag],5400):return False
            count+=1
            mark('pair','paired_episode_completed',episodes_completed=count,total_episodes=4,last_episode=tag)
    from comparison import summarize
    result=summarize(work,contract)
    write(root/'comparison.json',result)
    mark('pair','completed',episodes_completed=count,result=str(root/'comparison.json'),improvement_demonstrated=result['improvement_demonstrated'],automatic_patch_task_admitted=False)
    return True


def worker(kind):
    contract=freeze();folder=OUT/kind
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=folder/'budget.json'
        if ledger.exists():budget=read(ledger)
        else:
            budget=dict(started_epoch=time.time(),wall_seconds=contract['budgets']['asset_wall_seconds' if kind=='assets' else 'pair_wall_seconds'])
            write(ledger,budget)
        try:
            while not stopped(kind):
                if time.time()-budget['started_epoch']>=budget['wall_seconds']:
                    mark(kind,'budget_exhausted',budget_reset=False);return
                if (assets_tick(contract) if kind=='assets' else pair_tick(contract)):return
                time.sleep(15)
            mark(kind,'stopped')
        except BaseException as exc:
            mark(kind,'error',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['start-assets','start-pair','resume-assets','resume-pair','stop','status','brief','_worker','_preflight','_episode']);p.add_argument('--kind',choices=['assets','pair']);p.add_argument('--episode');a=p.parse_args()
    if a.command=='_worker':worker(a.kind);return
    if a.command=='_preflight':
        c=freeze();from episode import preflight
        preflight(c['policies']['primary'],OUT/'pair/preflight.json')
        from smpl_runtime import preflight as smpl_preflight
        write(OUT/'pair/smpl_preflight.json',smpl_preflight());print('Actual SMPL constructor verified; no GPU controls executed',flush=True);return
    if a.command=='_episode':
        c=freeze();from episode import execute
        folder=OUT/'pair/episodes'/a.episode;mode=read(folder/'comparison_mode.json');execute(c['task'],c['policies']['primary'],folder,'B' if mode['edited'] else 'A',mode=mode['mode']);return
    if a.command.startswith(('start-','resume-')):
        kind=a.command.split('-')[1]
        if a.command.startswith('resume-') and (OUT/kind/'STOP').exists():(OUT/kind/'STOP').unlink()
        print(json.dumps(start(kind),ensure_ascii=False,indent=2));return
    if a.command=='stop':
        for kind in ('assets','pair'):write(OUT/kind/'STOP',dict(reason='operator stop'))
        print('Stop requested; owned child groups will be stopped.');return
    data=status()
    if a.command=='brief':
        for kind,s in data.items():print('%s: %s | alive=%s | job=%s | completed=%s'%(kind,s['stage'],s['alive'],s.get('job','-'),s.get('episodes_completed','-')))
    else:print(json.dumps(data,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
