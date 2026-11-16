"""CPU-only publication checks. Never evaluates private simulation results."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/open-source'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args,env=None,cwd=None):
    begin=time.monotonic();r=subprocess.run(args,env=env,cwd=cwd or ROOT,capture_output=True,text=True)
    return dict(command=args,returncode=r.returncode,seconds=round(time.monotonic()-begin,3),stdout=r.stdout[-1600:],stderr=r.stderr[-1600:])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--skip-source-hashes',action='store_true');a=parser.parse_args()
    migration=read(DOC/'migration_manifest.json');copies=read(DOC/'copy_manifest.json')
    prune=read(DOC/'chain_prune_manifest.json') if (DOC/'chain_prune_manifest.json').exists() else {}
    archived={str(Path('integrations/chain-repair')/x['path']):x for x in prune.get('archived',[])} if prune.get('applied') else {}
    files={x['path'] for x in migration['changes']}|{x['destination'] for x in copies}
    errors=[];counts=dict(python=0,shell=0,json=0)
    for name in sorted(files):
        p=ROOT/name
        if name in archived and not p.exists():continue
        if p.suffix=='.py':
            counts['python']+=1
            try:ast.parse(p.read_text(),filename=name)
            except SyntaxError as e:errors.append(dict(path=name,error=str(e)))
        elif p.suffix=='.sh':
            counts['shell']+=1;r=command(['bash','-n',str(p)])
            if r['returncode']:errors.append(dict(path=name,error=r['stderr']))
        elif p.suffix=='.json':
            counts['json']+=1
            try:json.loads(p.read_text())
            except ValueError as e:errors.append(dict(path=name,error=str(e)))
    unchanged=None;missing=[];source_drift=[]
    if not a.skip_source_hashes:
        unchanged=True
        for row in copies:
            source=ROOT/row['source']
            current=sha(source)
            if current!=row['source_sha256']:
                unchanged=False;source_drift.append(dict(path=row['source'],snapshot_sha256=row['source_sha256'],current_sha256=current))
            if not (ROOT/row['destination']).is_file():
                archived_row=archived.get(row['destination'])
                archive=ROOT/prune.get('archive_root','')/archived_row['path'] if archived_row else None
                if archive is None or not archive.is_file() or sha(archive)!=archived_row['sha256']:missing.append(row['destination'])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    for args in [
        [sys.executable,'-B','-m','unittest','integrations._shared.test_paths','-v'],
        [sys.executable,'-B','integrations/scene-platform/cli.py','--help'],
        [sys.executable,'-B','integrations/scene-platform/cli.py','capabilities'],
        [sys.executable,'-B','integrations/scene-platform/cli.py','tools'],
        [sys.executable,'-B','integrations/novel-view-far-car-distillation/workflow.py','--help'],
        [sys.executable,'-B','integrations/chain-repair/phase26/bridge/agent_json/run_agent_json.py','--help'],
        [sys.executable,'-B','-m','unittest','discover','-s','integrations/scene-platform/tests','-p','test_*.py','-v'],
        ['bash','integrations/scene-platform/run.sh','capabilities'],
    ]:
        commands.append(command(args,env=env))
    # A real small checkout at a different path, with no historical outputs tree.
    with tempfile.TemporaryDirectory(prefix='hugsim-checkout-') as temp:
        repo=Path(temp)/'checkout with spaces';(repo/'integrations').mkdir(parents=True);(repo/'code').mkdir()
        for name in ['_shared','scene-platform']:
            shutil.copytree(ROOT/'integrations'/name,repo/'integrations'/name,ignore=shutil.ignore_patterns('__pycache__'))
        alternate=dict(env,HUGSIM_ROOT=str(repo))
        for key in ['HUGSIM_OUTPUT_ROOT','HUGSIM_EXTERNAL_ROOT','HUGSIM_CHAIN_CODE_ROOT']:alternate.pop(key,None)
        commands.append(command([sys.executable,'-B',str(repo/'integrations/scene-platform/cli.py'),'capabilities'],env=alternate,cwd='/tmp'))
        commands.append(command(['bash',str(repo/'integrations/scene-platform/run.sh'),'capabilities'],env=alternate,cwd='/tmp'))
    runtime_passed=not errors and not missing and all(c['returncode']==0 for c in commands)
    passed=runtime_passed and unchanged is not False
    report=dict(passed=passed,syntax_counts=counts,syntax_errors=errors,original_outputs_source_hashes_unchanged=unchanged,
                missing_source_copies=missing,copied_files_checked=len(copies),commands=commands,
                deliberately_archived_source_files=len(archived),
                minimum_runtime_checks_passed=runtime_passed,source_snapshot_drift=source_drift,
                gpu_executed=False,simulation_started=False,private_results_read=False,git_commands_executed=False)
    (DOC/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='commands'},indent=2))
    for c in commands:
        print(c['returncode'],' '.join(c['command']))
        if c['returncode']:print(c['stderr'])
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
