"""Move source-repository Git metadata and a failed download outside this checkout.

Explicitly scoped to the source repositories below. Environments, caches and
original outputs are left untouched. No git add/commit commands are run.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPOSITORIES=['code/HUGSIM','code/UniAD_SIM','code/VAD_SIM','code/NAVSIM',
              'code/scene_reconstruction','code/mmdetection3d','code/mmdetection3d-uniad-legacy',
              'integrations/sr-origin-main']
PLAN=ROOT/'docs/open-source/git_backup_plan.json'

def hashes(path):
    if path.is_file():return {'.':hashlib.sha256(path.read_bytes()).hexdigest()}
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}

def prepare():
    if PLAN.exists():return json.loads(PLAN.read_text())
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=ROOT.parent/'hugsim_git_backup'/stamp
    moves=[]
    for name in REPOSITORIES:
        source=ROOT/name/'.git'
        if source.exists():moves.append(dict(kind='git_metadata',source=name+'/.git',destination=str(backup/'git'/name/'.git')))
    fragment=ROOT/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth.part'
    if fragment.exists():moves.append(dict(kind='incomplete_download',source=str(fragment.relative_to(ROOT)),destination=str(backup/'download-fragments'/fragment.relative_to(ROOT))))
    plan=dict(schema='external_git_metadata_backup_v1',backup_root=str(backup),moves=moves,source_code_moved=False,outputs_modified=False)
    PLAN.write_text(json.dumps(plan,indent=2)+'\n');return plan

def apply():
    plan=prepare();report=[]
    for move in plan['moves']:
        source=ROOT/move['source'];target=Path(move['destination'])
        if target.exists():
            if source.exists():raise FileExistsError('Both source and backup exist: '+str(source))
            raise FileExistsError('Backup already exists; inspect prior receipt: '+str(target))
        before=hashes(source);target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['mv','--',str(source),str(target)],check=True)
        after=hashes(target)
        if before!=after:raise RuntimeError('Backup content hashes differ: '+str(target))
        report.append(dict(**move,files=len(before),content_hashes_verified=True))
    receipt=dict(**plan,completed=True,moved=report,no_commit_executed=True)
    (Path(plan['backup_root'])/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'docs/open-source/git_backup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.apply:apply()
    else:print(json.dumps(prepare(),indent=2))
