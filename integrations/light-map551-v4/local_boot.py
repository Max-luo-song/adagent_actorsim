"""Byte-identical legacy pure-Python startup support in one local archive."""

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

import hashlib,os,shutil,sys,tempfile,zipfile
from concurrent.futures import ThreadPoolExecutor
pass  # Path is provided by the portable boundary above
from common import ROOT,read,write,asset,sha,verify

def build(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    std=ROOT/'tools/pixi-uniad-legacy/.pixi/envs/default/lib/python3.8';site=std/'site-packages'
    paths=[]
    for directory,dirs,files in os.walk(std):
        dirs[:]=[d for d in dirs if d not in ('site-packages','__pycache__','lib-dynload')]
        for name in files:
            if name.endswith(('.py','.txt','.pickle','.cfg')):
                p=Path(directory)/name;paths.append((p,p.relative_to(std).as_posix()))
    for name in ['yapf','addict','packaging','terminaltables','six.py','termcolor.py','pyparsing.py','typing_extensions.py','zipp.py','importlib_metadata','toml']:
        p=site/name
        if p.is_file():paths.append((p,name))
        elif p.is_dir():paths.extend((f,f.relative_to(site).as_posix()) for f in p.rglob('*.py') if '__pycache__' not in f.parts)
    def load(item):
        p,name=item;data=p.read_bytes();return name,data,dict(source=str(p),member=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
    members=[]
    with zipfile.ZipFile(folder/'legacy_support.zip','x',compression=zipfile.ZIP_STORED) as z,ThreadPoolExecutor(max_workers=8) as pool:
        for name,data,receipt in pool.map(load,paths):z.writestr(name,data);members.append(receipt)
    result=dict(schema='byte_identical_legacy_pure_python_support_v1',archive=asset(folder/'legacy_support.zip'),members=members,
        binary_packages_changed=False,model_or_weights_changed=False,stdlib_abi='python3.8',
        purpose='Reduce shared-filesystem small-file reads before resident UniAD is ready; existing numerical runtime cache still activates unchanged')
    write(folder/'manifest.json',result);return result

def local_archive(manifest_path):
    data=read(manifest_path);verify(data['archive']);source=Path(data['archive']['path'])
    folder=Path(tempfile.gettempdir())/('hugsim-legacy-support-'+str(os.getuid()))/data['archive']['sha256'];folder.mkdir(parents=True,exist_ok=True)
    target=folder/'legacy_support.zip'
    if target.exists():
        if sha(target)!=data['archive']['sha256']:raise ValueError('Local legacy support changed')
    else:
        fd,name=tempfile.mkstemp(prefix='support-',dir=folder);os.close(fd);tmp=Path(name)
        try:
            shutil.copyfile(source,tmp)
            if sha(tmp)!=data['archive']['sha256']:raise ValueError('Legacy support copy failed')
            tmp.replace(target)
        finally:
            if tmp.exists():tmp.unlink()
    return str(target)
