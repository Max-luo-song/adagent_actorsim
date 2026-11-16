"""Bit-identical local copies of fixed runtime packages; no dependency installation."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import tempfile
import time
from startup_timing import atomic_json

BASE=Path(__file__).resolve().parents[1]
SITE=BASE/'tools/pixi-uniad-legacy/.pixi/envs/default/lib/python3.8/site-packages'
_ACTIVE_CHECKPOINT=None
ROOTS=('torch','torch-1.9.1+cu111.dist-info','numpy','numpy-1.20.0.dist-info','numpy.libs',
       'torchvision','torchvision-0.10.1+cu111.dist-info','torchvision.libs',
       'PIL','Pillow-9.5.0.dist-info','Pillow.libs','mmcv','mmcv_full-1.4.0.dist-info',
       'scipy','scipy-1.7.3.dist-info','scipy.libs','sklearn','scikit_learn-1.0.2.dist-info','scikit_learn.libs',
       'torchmetrics','torchmetrics-0.7.3.dist-info','mmdet','mmdet-2.14.0.dist-info','mmseg','mmsegmentation-0.14.1.dist-info')


class HashedReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
    def read(self,size=-1):
        value=self.stream.read(size);self.digest.update(value);return value


def digest(path):
    with Path(path).open('rb') as stream:
        reader=HashedReader(stream)
        while reader.read(1024*1024):pass
    return reader.digest.hexdigest()


def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:raise ValueError('Unsafe runtime bundle member')
    return path


def build(folder):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    archive=folder/'runtime.tar';members=[];started=time.monotonic()
    paths=[]
    for root in ROOTS:
        source=SITE/root
        if not source.is_dir():raise ValueError('Runtime package missing: '+root)
        for directory,dirs,files in os.walk(source):
            dirs[:]=sorted(d for d in dirs if d!='__pycache__')
            paths.extend(Path(directory)/name for name in sorted(files) if not name.endswith('.pyc'))
    def load(path):
        if path.is_symlink():raise ValueError('Unexpected symlink in runtime packages')
        before=path.stat();payload=None
        if before.st_size<=4*1024**2:
            payload=path.read_bytes();after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source runtime changed while copying')
        return path,before,payload
    with archive.open('wb') as raw:
        class Writer:
            def __init__(self):self.sha=hashlib.sha256()
            def write(self,value):self.sha.update(value);return raw.write(value)
        writer=Writer()
        with ThreadPoolExecutor(max_workers=12) as pool,tarfile.open(fileobj=writer,mode='w|') as tar:
            iterator=iter(paths);pending=deque()
            for _ in range(48):
                path=next(iterator,None)
                if path is not None:pending.append(pool.submit(load,path))
            while pending:
                path,before,payload=pending.popleft().result()
                relative=path.relative_to(SITE).as_posix();safe_name(relative)
                info=tarfile.TarInfo(relative);info.size=before.st_size;info.mode=before.st_mode&0o777;info.mtime=before.st_mtime
                if payload is None:
                    with path.open('rb') as stream:
                        hashed=HashedReader(stream);tar.addfile(info,hashed)
                    after=path.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source runtime changed while copying')
                    sha=hashed.digest.hexdigest()
                else:
                    tar.addfile(info,io.BytesIO(payload));sha=hashlib.sha256(payload).hexdigest()
                members.append(dict(path=relative,bytes=info.size,sha256=sha))
                path=next(iterator,None)
                if path is not None:pending.append(pool.submit(load,path))
                if len(members)%500==0:print('Bundled runtime files',len(members),'of',len(paths),flush=True)
        archive_sha=writer.sha.hexdigest()
    manifest=dict(schema='uniad_runtime_bundle_v1',archive=str(archive),archive_sha256=archive_sha,
        source_site=str(SITE),roots=list(ROOTS),members=members,total_bytes=sum(item['bytes'] for item in members),
        build_seconds=time.monotonic()-started,python_abi='cp38',training=False,
        source_records={root:digest(SITE/root/'RECORD') for root in ROOTS if root.endswith('.dist-info')})
    atomic_json(folder/'manifest.json',manifest)
    print(json.dumps({key:manifest[key] for key in ('archive','total_bytes','build_seconds')}),flush=True)
    return manifest


def extract_verified(archive,manifest,target):
    target=Path(target);expected={item['path']:item for item in manifest['members']}
    if len(expected)!=len(manifest['members']):raise ValueError('Duplicate manifest names')
    seen=set()
    with Path(archive).open('rb') as raw:
        reader=HashedReader(raw)
        with tarfile.open(fileobj=reader,mode='r|') as tar:
            for member in tar:
                safe_name(member.name)
                if not member.isfile() or member.name not in expected or member.name in seen:raise ValueError('Unexpected runtime member')
                item=expected[member.name]
                if member.size!=item['bytes']:raise ValueError('Runtime size mismatch')
                path=target/member.name;path.parent.mkdir(parents=True,exist_ok=True)
                sha=hashlib.sha256()
                with tar.extractfile(member) as source,path.open('wb') as dest:
                    while True:
                        block=source.read(1024*1024)
                        if not block:break
                        dest.write(block);sha.update(block)
                if sha.hexdigest()!=item['sha256']:raise ValueError('Runtime file SHA mismatch: '+member.name)
                seen.add(member.name)
        while reader.read(1024*1024):pass
        if reader.digest.hexdigest()!=manifest['archive_sha256']:raise ValueError('Runtime archive SHA mismatch')
    if seen!=set(expected):raise ValueError('Incomplete runtime bundle')


def prepare(manifest_path):
    started=time.monotonic();manifest_path=Path(manifest_path).resolve()
    if BASE not in manifest_path.parents:raise ValueError('Runtime manifest must stay inside project')
    manifest=json.loads(manifest_path.read_text())
    if manifest['schema']!='uniad_runtime_bundle_v1' or tuple(manifest['roots'])!=ROOTS or manifest['source_site']!=str(SITE):raise ValueError('Unexpected runtime bundle')
    archive=Path(manifest['archive']).resolve()
    if BASE not in archive.parents or not archive.is_file():raise ValueError('Runtime archive must stay inside project')
    for root,sha in manifest['source_records'].items():
        if root not in ROOTS or digest(SITE/root/'RECORD')!=sha:raise ValueError('Installed runtime metadata changed')
    parent=Path(tempfile.gettempdir())/('hugsim-uniad-runtime-'+str(os.getuid()));parent.mkdir(mode=0o700,exist_ok=True)
    target=parent/manifest['archive_sha256'];hit=False
    with (parent/'cache.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        complete=target/'complete.json'
        if complete.is_file():
            data=json.loads(complete.read_text())
            if data.get('archive_sha256')!=manifest['archive_sha256']:raise ValueError('Invalid cached runtime')
            for item in manifest['members']:
                safe_name(item['path'])
                if digest(target/item['path'])!=item['sha256']:raise ValueError('Cached runtime file changed')
            hit=True
        else:
            if shutil.disk_usage(parent).free<manifest['total_bytes']+256*1024**2:raise RuntimeError('Insufficient local disk for runtime cache')
            temporary=Path(tempfile.mkdtemp(prefix='staging-',dir=parent))
            try:
                extract_verified(archive,manifest,temporary)
                atomic_json(temporary/'complete.json',dict(archive_sha256=manifest['archive_sha256']))
                temporary.rename(target)
            finally:
                if temporary.exists():shutil.rmtree(temporary)
    return dict(site_packages=str(target),cache_hit=hit,prepare_seconds=time.monotonic()-started,
                verified_bytes=manifest['total_bytes'],archive_sha256=manifest['archive_sha256'],manifest=str(manifest_path))


def checkpoint_path():
    return _ACTIVE_CHECKPOINT


def prepare_checkpoint(expected,folder):
    source=BASE/'code/UniAD_SIM/ckpts/uniad_base_e2e.pth'
    if set(expected)!={'bytes','sha256','mtime_ns'}:raise ValueError('Invalid frozen checkpoint identity')
    before=source.stat()
    if (before.st_size,before.st_mtime_ns)!=(expected['bytes'],expected['mtime_ns']):raise ValueError('Frozen checkpoint changed; refresh runtime configuration')
    started=time.monotonic();folder=Path(folder);target=folder/('checkpoint-'+expected['sha256']+'.pth');hit=False
    with (folder/'checkpoint.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.is_file():
            if digest(target)!=expected['sha256']:raise ValueError('Cached checkpoint changed')
            hit=True
        else:
            if shutil.disk_usage(folder).free<expected['bytes']+256*1024**2:raise RuntimeError('Insufficient local disk for frozen checkpoint')
            fd,name=tempfile.mkstemp(prefix='checkpoint-staging-',dir=folder);temporary=Path(name)
            try:
                sha=hashlib.sha256()
                with source.open('rb') as src,os.fdopen(fd,'wb') as dst:
                    while True:
                        block=src.read(1024*1024)
                        if not block:break
                        dst.write(block);sha.update(block)
                after=source.stat()
                if sha.hexdigest()!=expected['sha256'] or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Frozen checkpoint copy failed integrity check')
                temporary.rename(target)
            finally:
                if temporary.exists():temporary.unlink()
    return dict(path=str(target),sha256=expected['sha256'],cache_hit=hit,prepare_seconds=time.monotonic()-started)


def activate_configured():
    """Common worker bootstrap policy, independent of scenes and model inputs."""
    import sys
    global _ACTIVE_CHECKPOINT
    config=BASE/'waymo23-native-import/configs/uniad_runtime_cache.json'
    if not config.is_file():return None
    settings=json.loads(config.read_text())
    if set(settings)!={'schema','enabled','manifest','checkpoint'} or settings['schema']!='uniad_runtime_cache_v1' or not isinstance(settings['enabled'],bool):raise ValueError('Invalid runtime cache configuration')
    if not settings['enabled']:return None
    if 'numpy' in sys.modules or 'torch' in sys.modules:raise RuntimeError('Runtime cache must activate before numerical imports')
    result=prepare(settings['manifest'])
    result['checkpoint']=prepare_checkpoint(settings['checkpoint'],Path(result['site_packages']).parent)
    _ACTIVE_CHECKPOINT=result['checkpoint']['path']
    sys.path.insert(0,result['site_packages'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',required=True)
    build(parser.parse_args().build)
