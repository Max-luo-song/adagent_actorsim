"""Conservative source dependency closure; archive unreferenced snapshot code.

Only implementation/configuration files are inspected. Never reads experiment
results or mutates original outputs. Unknown dynamic imports retain all matching
module implementations; kept package trees exclude unreferenced diagnostic/tests.
"""
import argparse
import ast
from collections import Counter,defaultdict,deque
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CHAIN=ROOT/'integrations/chain-repair'
CR=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
REPORT=ROOT/'docs/open-source/chain_prune_manifest.json'
ARCHIVE=ROOT/'.local/chain-repair-unreferenced-20261116'
EXCLUDE={'diagnostics','backups','legacy_tests','tests','test','dev','docs','camera_id_recovery','smpl_runtime_recovery'}
SKIP={'.git','.pixi','.teacher-env','.venv','venv','build','.cache','__pycache__','outputs','node_modules'}

def files(root):
    import os
    for directory,dirs,names in os.walk(root):
        dirs[:]=[n for n in dirs if n not in SKIP and not Path(directory,n).is_symlink()]
        for name in names:
            p=Path(directory,name)
            if p.is_file() and not p.is_symlink():yield p

def evaluate(node,env):
    try:
        if isinstance(node,ast.Name):return env.get(node.id)
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.BinOp):
            a,b=evaluate(node.left,env),evaluate(node.right,env)
            if isinstance(node.op,ast.Div) and isinstance(a,(Path,str)) and isinstance(b,str):return Path(a)/b
            if isinstance(node.op,ast.Add) and isinstance(a,str) and isinstance(b,str):return a+b
        if isinstance(node,ast.Attribute):
            a=evaluate(node.value,env)
            if node.attr=='parent' and isinstance(a,Path):return a.parent
            if node.attr=='parents' and isinstance(a,Path):return a.parents
            if node.attr in {'ROOT','REPO','BASE'}:return ROOT
            if node.attr=='P26':return CR/'phase26'
            if node.attr=='CR':return CR
        if isinstance(node,ast.Subscript):
            a,b=evaluate(node.value,env),evaluate(node.slice,env)
            if a is not None and isinstance(b,int):return a[b]
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else None
            if name in {'Path','_HugsimPath','_hugsim_expand','_hugsim_path','str'} and node.args:
                value=evaluate(node.args[0],env)
                if isinstance(value,(str,Path)):
                    value=str(value).replace('${HUGSIM_ROOT}',str(ROOT))
                    return Path(value) if name in {'Path','_HugsimPath'} else value
            if isinstance(node.func,ast.Attribute) and node.func.attr in {'resolve','absolute','expanduser'}:return evaluate(node.func.value,env)
    except (TypeError,ValueError,IndexError):return None
    return None

def closure():
    all_files=list(files(CHAIN));by_path={p.relative_to(CHAIN).as_posix():p for p in all_files}
    modules=defaultdict(set)
    for rel,p in by_path.items():
        if p.suffix=='.py':modules[p.stem].add(rel)
    keep={};queue=deque();scanned=set();dynamic=[];targets_processed=set()
    def retain(rel,reason):
        if rel in by_path and rel not in keep:keep[rel]=reason;queue.append(by_path[rel])
    def target(value,reason):
        if not isinstance(value,(str,Path)):return
        value=str(value).replace('${HUGSIM_ROOT}',str(ROOT))
        if value.startswith(str(CR)):rel=value[len(str(CR)):].lstrip('/')
        elif value.startswith(str(CHAIN)):rel=value[len(str(CHAIN)):].lstrip('/')
        else:return
        rel=rel.rstrip('/')
        if rel in targets_processed:return
        targets_processed.add(rel)
        if rel in by_path:retain(rel,reason);return
        # Bare phase constants are aliases, not evidence that all phases are used.
        if not rel or '/' not in rel:return
        prefix=rel+'/'
        for name in by_path:
            if name.startswith(prefix) and not any(x in EXCLUDE for x in Path(name[len(prefix):]).parts):retain(name,reason)
    def inspect(path):
        if path in scanned:return
        scanned.add(path)
        if path.suffix not in {'.py','.sh','.json','.yaml','.yml','.toml','.cfg','.ini'}:return
        text=path.read_text(errors='replace');reason=str(path.relative_to(ROOT))
        # Literal full paths and phase-qualified resources, including shell paths.
        for match in re.finditer(r'\b(phase\d+/[A-Za-z0-9_.+/-]+)',text):target(CR/match[1].rstrip('/.'),reason)
        if path.suffix!='.py':return
        try:tree=ast.parse(text)
        except SyntaxError:return
        logical=CR/path.relative_to(CHAIN) if path.is_relative_to(CHAIN) else path
        env={'ROOT':ROOT,'REPO':ROOT,'BASE':ROOT,'CR':CR,'P26':CR/'phase26','HERE':logical.parent,'LOCAL':logical.parent,'__file__':str(logical)}
        for node in tree.body:
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                value=evaluate(node.value,env)
                names=node.targets if isinstance(node,ast.Assign) else [node.target]
                for name in names:
                    if isinstance(name,ast.Name) and value is not None:env[name.id]=value
        for node in ast.walk(tree):
            if isinstance(node,(ast.BinOp,ast.Call,ast.Constant)):target(evaluate(node,env),reason)
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
                if isinstance(node,ast.ImportFrom) and node.module is None:names.extend(n.name for n in node.names)
                for name in names:
                    head=name.split('.')[0]
                    if not head:continue
                    candidates=[]
                    if path.is_relative_to(CHAIN):
                        directory=path.parent
                        for parent in [directory,*directory.parents]:
                            if not parent.is_relative_to(CHAIN):break
                            for candidate in [parent/(head+'.py'),parent/head/'__init__.py']:
                                relative=candidate.relative_to(CHAIN).as_posix()
                                if relative in by_path:candidates.append(relative)
                            if candidates:break
                    if not candidates:candidates=list(modules.get(head,()))
                    for rel in candidates:retain(rel,reason+' import '+head)
    for root in [ROOT/'code',ROOT/'integrations',ROOT/'waymo23-native-import',ROOT/'tools']:
        for p in files(root):
            if not p.is_relative_to(CHAIN):inspect(p)
    while queue:inspect(queue.popleft())
    # Keep notices and requirements adjacent to selected runtime packages.
    notices=defaultdict(list)
    for name in by_path:
        q=Path(name)
        if q.name.startswith(('LICENSE','NOTICE','COPYING','requirements')):notices[q.parent].append(name)
    for rel in list(keep):
        p=Path(rel)
        for ancestor in [p.parent,*p.parent.parents]:
            if str(ancestor)=='.':break
            for name in notices.get(ancestor,()):retain(name,'notice/dependency metadata for '+rel)
    retain('README.md','source bundle documentation')
    archive=sorted(set(by_path)-set(keep))
    return by_path,keep,archive,len(scanned)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');parser.add_argument('--apply-reviewed-plan',action='store_true');a=parser.parse_args()
    if a.apply_reviewed_plan:
        report=json.loads(REPORT.read_text())
        if report['applied']:raise ValueError('Reviewed plan is already applied')
        if ARCHIVE.exists():raise FileExistsError(ARCHIVE)
        for row in report['archived']:
            p=CHAIN/row['path']
            if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Reviewed source changed: '+row['path'])
        for row in report['archived']:
            source=CHAIN/row['path'];target=ARCHIVE/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(source),str(target))
        for p in sorted([p for p in CHAIN.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
            try:p.rmdir()
            except OSError:pass
        report['applied']=True;report['reviewed_source_hashes_verified']=True
        report['after_du_sh']=subprocess.check_output(['du','-sh',str(CHAIN)],text=True).strip()
        REPORT.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in {'kept','archived'}},indent=2));return
    by,keep,archive,scanned=closure()
    before=sum(p.stat().st_size for p in by.values());removed=sum(by[p].stat().st_size for p in archive)
    report=dict(schema='chain_source_prune_v1',applied=a.apply,method='Conservative static path + import closure from all public code/integrations/tools/native adapters; preserve unknown matching imports',
                before_files=len(by),kept_files=len(keep),archived_files=len(archive),before_bytes=before,kept_bytes=before-removed,archived_bytes=removed,
                source_files_scanned=scanned,archive_root=str(ARCHIVE.relative_to(ROOT)),
                before_extensions=dict(Counter(p.suffix or '<none>' for p in by.values())),
                after_extensions=dict(Counter(Path(p).suffix or '<none>' for p in keep)),
                kept=[dict(path=p,reason=keep[p]) for p in sorted(keep)],
                archived=[dict(path=p,sha256=hashlib.sha256(by[p].read_bytes()).hexdigest()) for p in archive],
                original_outputs_modified=False,dynamic_dependency_certification=False)
    if a.apply:
        if ARCHIVE.exists():raise FileExistsError('Existing archive retained: '+str(ARCHIVE))
        for name in archive:
            destination=ARCHIVE/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(by[name]),destination)
        for p in sorted([p for p in CHAIN.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
            try:p.rmdir()
            except OSError:pass
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'kept','archived'}},indent=2))

if __name__=='__main__':main()
