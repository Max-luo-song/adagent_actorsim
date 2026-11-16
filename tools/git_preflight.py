"""Read-only Git candidate audit; optionally extend ignore rules for oversize files.

Does not stage, commit, follow symlinks, print secret values, or inspect ignored
experiment results. Rule-based secret checks are detection, not a proof of absence.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/open-source'
LIMIT=50_000_000
PATTERNS={
 'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
 'service_api_key':re.compile(rb'\bsk-(?:proj-|ant-|svcacct-)?[A-Za-z0-9_-]{20,}\b'),
 'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
 'aws_access_id':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'huggingface_token':re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
 'slack_token':re.compile(rb'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
 'literal_credential':re.compile(rb'''(?i)(?:api[_-]?key|api[_-]?token|secret[_-]?key|access[_-]?key|client[_-]?secret|aws_secret_access_key|password|\bAK|\bSK)\s*[=:]\s*["']([A-Za-z0-9_+/.=-]{12,})["']'''),
 'dotenv_credential':re.compile(rb'(?m)^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:API_KEY|ACCESS_KEY_ID|SECRET_ACCESS_KEY|PASSWORD|API_TOKEN|\bAK|\bSK)[ \t]*=[ \t]*([A-Za-z0-9_+/=-]{12,})[ \t]*$'),
 'bearer_token':re.compile(rb'(?i)\bBearer[ \t]+([A-Za-z0-9._~+/=-]{20,})'),
}
PLACEHOLDER=re.compile(rb'(?i)^(?:(?:sk-|hf_)?(?:example|placeholder|dummy|fake|changeme|replace|your[_-]|test[_-])|test$|AKIAIOSFODNN7EXAMPLE$|[xX0]{12,}$|123456|abcdef)')
PREFIXES={'private_key':(b'PRIVATE KEY',),'service_api_key':(b'sk-',),'github_token':(b'ghp_',b'gho_',b'ghu_',b'ghs_',b'ghr_',b'github_pat_'),
          'aws_access_id':(b'AKIA',b'ASIA'),'huggingface_token':(b'hf_',),'slack_token':(b'xox',)}

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)

def candidates():
    return sorted(set(p.decode('utf-8') for p in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if p))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply-size-ignores',action='store_true');a=p.parse_args()
    names=candidates();oversize=[]
    for name in names:
        f=ROOT/name
        if f.is_file() and not f.is_symlink() and f.lstat().st_size>LIMIT:oversize.append(dict(path=name,bytes=f.lstat().st_size))
    if a.apply_size_ignores and oversize:
        with (ROOT/'.gitignore').open('a') as stream:
            stream.write('\n# Additional assets detected by the strict 50 MB preflight.\n')
            for row in oversize:stream.write('/'+row['path'].replace(' ','\\ ')+'\n')
        names=candidates()
    findings=[];placeholders=[];files=[];bytes_scanned=0;symlinks=[]
    for name in names:
        f=ROOT/name;size=f.lstat().st_size
        files.append(dict(path=name,bytes=size))
        if f.is_symlink():symlinks.append(name);continue
        # No candidate exceeding the limit is opened or allowed to pass.
        if size>LIMIT:continue
        raw=f.read_bytes();bytes_scanned+=len(raw)
        # Scan credential signatures even in binary files. Generic assignments
        # are limited to text to avoid false matches in scientific/binary data.
        text=b'\0' not in raw[:8192]
        for kind,pattern in PATTERNS.items():
            if kind in {'literal_credential','dotenv_credential','bearer_token'} and not text:continue
            if kind in PREFIXES and not any(prefix in raw for prefix in PREFIXES[kind]):continue
            for match in pattern.finditer(raw):
                token=match.group(1) if kind in {'literal_credential','dotenv_credential','bearer_token'} else match.group(0)
                item=dict(path=name,line=raw.count(b'\n',0,match.start())+1,kind=kind)
                line=raw[raw.rfind(b'\n',0,match.start())+1:raw.find(b'\n',match.end()) if b'\n' in raw[match.end():] else len(raw)]
                env_name = kind=='literal_credential' and re.search(rb'\benv_var_[a-z_]+\s*=',line) and re.fullmatch(rb'[a-z][a-z0-9_]+',token)
                if PLACEHOLDER.search(token) or env_name:placeholders.append(dict(**item,reason='environment variable name' if env_name else 'explicit placeholder/test fixture'))
                else:findings.append(item)
    metadata=[]
    for start in [ROOT/'code',ROOT/'integrations',ROOT/'waymo23-native-import',ROOT/'tools']:
        for directory,dirs,names_here in os.walk(start):
            dirs[:]=[d for d in dirs if d not in {'.pixi','.teacher-env','.venv','venv','env','build','.cache','__pycache__','node_modules','outputs'}]
            path=Path(directory)/'.git'
            if path.exists():metadata.append(str(path.relative_to(ROOT)));dirs[:]=[d for d in dirs if d!='.git']
    untracked_status=git('status','--short','--untracked-files=all','-z')
    branch=git('branch','--show-current').decode().strip()
    staged=git('diff','--cached','--name-only','-z')
    tracked=git('ls-files','-z')
    current_large=[row for row in files if row['bytes']>LIMIT]
    report=dict(schema='github_ingestion_preflight_v1',branch=branch,untracked_file_count=len(files),
                total_logical_bytes=sum(row['bytes'] for row in files),total_MiB=round(sum(row['bytes'] for row in files)/1024**2,3),
                total_allocated_bytes=sum((ROOT/row['path']).lstat().st_blocks*512 for row in files),
                allocated_MiB=round(sum((ROOT/row['path']).lstat().st_blocks*512 for row in files)/1024**2,3),
                status_entry_count=sum(bool(x) for x in untracked_status.split(b'\0')),status_matches_untracked_count=sum(bool(x) for x in untracked_status.split(b'\0'))==len(files),
                threshold_bytes=LIMIT,oversized_candidates=current_large,secret_findings=findings,recognized_placeholders=placeholders,
                secret_values_recorded=False,secret_scan_bytes=bytes_scanned,secret_scan_method='Known key/private-key signatures and literal credential assignments; no network validation',
                source_nested_git=metadata,symlinks=symlinks,staged_file_count=sum(bool(x) for x in staged.split(b'\0')),tracked_file_count=sum(bool(x) for x in tracked.split(b'\0')),
                oversized_ignore_additions=oversize if a.apply_size_ignores else [],
                file_extensions=dict(Counter(Path(row['path']).suffix or '<none>' for row in files)),
                passed=branch=='main' and not current_large and not findings and not metadata and not staged and not tracked,
                no_commit_executed=True)
    DOC.mkdir(exist_ok=True,parents=True)
    initial_names=set(names)
    generated={str((DOC/name).relative_to(ROOT)) for name in ['git_preflight.json','git_candidate_files.txt','git_candidate_sizes.tsv','git_status_short.txt']}
    # Include the audit files themselves. Iterate until their own byte counts and
    # the total agree, rather than reporting the pre-report filesystem snapshot.
    for _ in range(10):
        (DOC/'git_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
        (DOC/'git_candidate_files.txt').write_text('\n'.join(row['path'] for row in files)+'\n')
        (DOC/'git_candidate_sizes.tsv').write_text('bytes\tpath\n'+''.join(str(row['bytes'])+'\t'+row['path']+'\n' for row in files))
        status=git('status','--short','--untracked-files=all','-z')
        (DOC/'git_status_short.txt').write_text(status.replace(b'\0',b'\n').decode('utf-8'))
        now=candidates();updated=[dict(path=name,bytes=(ROOT/name).lstat().st_size) for name in now]
        total=sum(row['bytes'] for row in updated)
        unexpected=sorted(set(now)-initial_names-generated)
        stable=updated==files and report['total_logical_bytes']==total
        files=updated
        report.update(untracked_file_count=len(files),total_logical_bytes=total,total_MiB=round(total/1024**2,3),
                      total_allocated_bytes=sum((ROOT/row['path']).lstat().st_blocks*512 for row in files),
                      allocated_MiB=round(sum((ROOT/row['path']).lstat().st_blocks*512 for row in files)/1024**2,3),
                      status_entry_count=sum(bool(x) for x in status.split(b'\0')),
                      status_matches_untracked_count=sum(bool(x) for x in status.split(b'\0'))==len(files),
                      oversized_candidates=[row for row in files if row['bytes']>LIMIT],
                      unexpected_files_added_during_scan=unexpected)
        report['passed']=report['passed'] and not unexpected and not report['oversized_candidates']
        if stable:break
    else:raise RuntimeError('Candidate tree/report byte counts did not stabilize')
    print(json.dumps({k:v for k,v in report.items() if k not in {'recognized_placeholders','file_extensions','symlinks'}},indent=2))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
