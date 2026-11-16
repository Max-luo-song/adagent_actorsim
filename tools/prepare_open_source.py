"""One-shot, auditable source-only copy and portable-path migration.

Never writes inside outputs, never invokes git, never opens private result JSON.
Input file inventories are supplied explicitly to avoid walking environments.
"""
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
LEGACY=ROOT/'outputs/gpu_test_service/jobs/unified-open-closed-172-20261022-v4/chain_repair'
DEST=ROOT/'integrations/chain-repair'
REPORT=ROOT/'docs/open-source'
SOURCE_SUFFIX={'.py','.sh','.c','.cc','.cpp','.cu','.h','.hpp','.cuh','.yaml','.yml','.toml','.cfg','.ini','.in'}
SKIP_PARTS={'.git','.pixi','.teacher-env','.cache','.venv','venv','node_modules','site-packages','__pycache__'}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def root_token(text):
    text=text.replace(str(ROOT),'${HUGSIM_ROOT}')
    text=re.sub(r'/inspire/(?:hdd3/project/intelligent-driving-agent|hdd/global_user)/[^/\s\"\']+/', '${HUGSIM_EXTERNAL_ROOT}/',text)
    return re.sub(r'/inspire/([^\s\"\'<>]+)',r'${HUGSIM_EXTERNAL_ROOT}/\1',text)

def bootstrap():
    return ('\n# Portable source/data boundary (no global pathlib patch).\n'
      'import sys as _hugsim_sys\nfrom pathlib import Path as _HugsimNativePath\n'
      '_hugsim_repository = next((p for p in _HugsimNativePath(__file__).resolve().parents '
      'if (p / "integrations/_shared/paths.py").is_file()), None)\n'
      'if _hugsim_repository is None:\n    raise RuntimeError("Repository root missing: integrations/_shared/paths.py")\n'
      'if str(_hugsim_repository) not in _hugsim_sys.path:\n    _hugsim_sys.path.insert(0, str(_hugsim_repository))\n'
      'from integrations._shared.paths import Path as _HugsimPath, portable_path as _hugsim_path, expand_paths as _hugsim_expand\n')

def python_text(text):
    if '# Portable source/data boundary (no global pathlib patch).' in text:
        return text
    tree=ast.parse(text);lines=text.splitlines(keepends=True);starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    # AST columns are UTF-8 byte offsets; preserve non-ASCII source exactly.
    def pos(line,column):return starts[line-1]+len(lines[line-1].encode()[:column].decode())
    def span(n):return pos(n.lineno,n.col_offset),pos(n.end_lineno,n.end_col_offset)
    edits=[];path_names=[];json_modules=set();json_functions=set();path_modules=set()
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    docnodes=set()
    for n in ast.walk(tree):
        if isinstance(n,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant):docnodes.add(n.body[0].value)
    for n in ast.walk(tree):
        if isinstance(n,ast.ImportFrom) and n.module=='pathlib':
            found=[a for a in n.names if a.name=='Path']
            if found:
                path_names.extend(a.asname or a.name for a in found)
                others=[a for a in n.names if a.name!='Path'];a,b=span(n)
                replacement='from pathlib import '+', '.join(x.name+(' as '+x.asname if x.asname else '') for x in others) if others else 'pass  # Path is provided by the portable boundary above'
                edits.append((a,b,replacement))
        elif isinstance(n,ast.Import):
            for a in n.names:
                if a.name=='json':json_modules.add(a.asname or a.name)
                if a.name=='pathlib':path_modules.add(a.asname or a.name)
        elif isinstance(n,ast.ImportFrom) and n.module=='json':
            json_functions.update(a.asname or a.name for a in n.names if a.name in {'load','loads'})
    for n in ast.walk(tree):
        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id in path_modules and n.attr=='Path':
            a,b=span(n);edits.append((a,b,'_HugsimPath'))
        if isinstance(n,ast.Call):
            f=n.func
            is_json=(isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id in json_modules and f.attr in {'load','loads'}) or (isinstance(f,ast.Name) and f.id in json_functions)
            if is_json:
                a,b=span(n);edits.extend([(a,a,'_hugsim_expand('),(b,b,')')])
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and '/inspire/' in n.value and not isinstance(parents.get(n),ast.JoinedStr):
            a,b=span(n);value=root_token(n.value)
            edits.append((a,b,repr(value) if n in docnodes else '_hugsim_expand('+repr(value)+')'))
        elif isinstance(n,ast.JoinedStr) and any(isinstance(x,ast.Constant) and isinstance(x.value,str) and '/inspire/' in x.value for x in n.values):
            a,b=span(n);literal=root_token(text[a:b]);edits.append((a,b,'_hugsim_expand('+literal+')'))
    if not edits:return text
    # Reject nested whole replacements instead of silently damaging syntax.
    intervals=sorted((a,b) for a,b,v in edits if a!=b)
    if any(b>c for (a,b),(c,d) in zip(intervals,intervals[1:])):raise ValueError('Overlapping path replacements')
    end=0
    body=tree.body
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):end=body[0].end_lineno
    for n in body:
        if isinstance(n,ast.ImportFrom) and n.module=='__future__':end=max(end,n.end_lineno)
    if not end:
        for i,line in enumerate(lines[:2]):
            if line.startswith('#!') or re.search(r'coding[:=]',line):end=i+1
    injected=bootstrap()+''.join(f'{name} = _HugsimPath\n' for name in dict.fromkeys(path_names))+'\n'
    edits.append((starts[end],starts[end],injected))
    # Existing edits at identical coordinates must keep wrapper order stable.
    for a,b,value in sorted(edits,key=lambda e:(e[0],e[1]),reverse=True):text=text[:a]+value+text[b:]
    ast.parse(text)
    return text

def shell_text(text):
    if '# Locate the repository without an installation-specific absolute path.' in text:
        return text
    text=root_token(text)
    # Existing root computations have different depths in phase snapshots.
    text=re.sub(r'(?m)^ROOT=.*$', 'ROOT="${HUGSIM_ROOT}"',text)
    text=re.sub(r'(?m)^PY=.*$', 'PY="${HUGSIM_PYTHON}"',text)
    # Dispatch scripts from the copied implementation tree, data arguments stay put.
    text=re.sub(r'"(\$(?:\{?[A-Za-z_][A-Za-z_0-9]*\}?)[^"\n;]*\.(?:py|sh))"',lambda m:'"$(hugsim_source_path "'+m[1]+'")"',text)
    lines=text.splitlines(keepends=True);index=1 if lines and lines[0].startswith('#!') else 0
    block=('\n# Locate the repository without an installation-specific absolute path.\n'
      '_hugsim_boot="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
      'while [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" && "$_hugsim_boot" != / ]]; do _hugsim_boot="$(dirname "$_hugsim_boot")"; done\n'
      'if [[ ! -f "$_hugsim_boot/integrations/_shared/env.sh" ]]; then echo "HUGSIM repository root not found" >&2; exit 2; fi\n'
      'source "$_hugsim_boot/integrations/_shared/env.sh"\nunset _hugsim_boot\n')
    lines.insert(index,block);return ''.join(lines)

def config_text(text,suffix):
    text=root_token(text)
    if suffix in {'.yaml','.yml'}:
        for name in ['HUGSIM_ROOT','HUGSIM_OUTPUT_ROOT','HUGSIM_EXTERNAL_ROOT']:
            default='.' if name=='HUGSIM_ROOT' else ('./outputs' if name=='HUGSIM_OUTPUT_ROOT' else './assets/external')
            text=text.replace('${'+name+'}','${oc.env:'+name+','+default+'}')
    return text

def main():
    global _migration_lock
    _migration_lock=open('/tmp/hugsim-open-source-migration.lock','a')
    fcntl.flock(_migration_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-list',type=Path,required=True);parser.add_argument('--chain-list',type=Path,required=True);args=parser.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');backup=Path('/tmp')/('hugsim-open-source-'+stamp);backup.mkdir()
    REPORT.mkdir(parents=True,exist_ok=True)
    existing=[ROOT/p for p in args.source_list.read_text().splitlines()]
    chain=[ROOT/p for p in args.chain_list.read_text().splitlines()]
    copies=[];changed=[];errors=[]
    for src in chain:
        config=src.suffix=='.json' and (any(x in src.parts for x in ['schemas','configs','examples']) or src.name=='catalog.json' or src.name.endswith(('_command.json','.schema.json')))
        extra=src.name.startswith(('LICENSE','COPYING','requirements')) or src.name in {'CMakeLists.txt','setup.cfg','MANIFEST.in'}
        if src.suffix not in SOURCE_SUFFIX and not config and not extra:continue
        if src.is_symlink() or any(part in SKIP_PARTS for part in src.parts):continue
        if src.stat().st_size>50*1024**2:errors.append(dict(path=str(src.relative_to(ROOT)),reason='source file exceeds 50 MiB'));continue
        dst=DEST/src.relative_to(LEGACY);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if digest(dst)!=digest(src):raise FileExistsError('Existing destination differs: '+str(dst))
        else:shutil.copy2(src,dst)
        copies.append(dict(source=str(src.relative_to(ROOT)),destination=str(dst.relative_to(ROOT)),source_sha256=digest(src)))
    (REPORT/'copy_manifest.json').write_text(json.dumps(copies,indent=2)+'\n')
    print('Copied source/config files:',len(copies),flush=True)
    candidates=existing+[ROOT/x['destination'] for x in copies]
    for index,p in enumerate(candidates):
        if p.is_symlink() or not p.is_file() or p.suffix not in {'.py','.sh','.json','.yaml','.yml','.toml','.cfg','.ini','.md','.txt'}:continue
        if 'integrations/_shared' in str(p) or p.stat().st_size>2_000_000:continue
        text=p.read_text(errors='strict');moved=p.is_relative_to(DEST)
        if p.suffix=='.py':
            # Runtime code under integrations may load portable JSON dynamically.
            selected=moved or p.is_relative_to(ROOT/'integrations') or 'chain_repair' in text or '/inspire/' in text or 'pathlib' in text
            if not selected:continue
            try:new=python_text(text)
            except (SyntaxError,ValueError) as error:errors.append(dict(path=str(p.relative_to(ROOT)),reason=str(error)));continue
        elif p.suffix=='.sh':new=shell_text(text) if moved or p.is_relative_to(ROOT/'integrations') or '/inspire/' in text or 'chain_repair' in text else text
        else:new=config_text(text,p.suffix) if '/inspire/' in text else text
        if new!=text:
            rel=p.relative_to(ROOT);saved=backup/rel;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,saved)
            before=digest(p);p.write_text(new);changed.append(dict(path=str(rel),before_sha256=before,after_sha256=digest(p)))
        if index%500==0:print('Processed',index,'/',len(candidates),flush=True)
    report=dict(schema='open_source_migration_v1',copied_count=len(copies),modified_count=len(changed),backup=str(backup),changes=changed,errors=errors,outputs_modified=False,git_commands_executed=False)
    (REPORT/'migration_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (REPORT/'changed_files.txt').write_text('\n'.join(x['path'] for x in changed)+'\n')
    (REPORT/'copied_files.txt').write_text('\n'.join(x['destination'] for x in copies)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'changes','errors'}},indent=2),flush=True);print('Transform exceptions',len(errors),flush=True)

if __name__=='__main__':main()
