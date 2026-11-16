"""Download official weather weights without touching UniAD or starting GPU work."""

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

HERE = Path(__file__).resolve().parent
REVISION = '5774135e8892e5ee658a854a4c7c3b349ced28e2'
WEATHER_SHA256 = '742c6eeebe50d455cef7df1080e1a811baa9dafa52426808d557897e5505c645'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def publish_link(target, source):
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f'refusing to replace existing path: {target}')
    else:
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=HERE.parents[1]/'framework_repo/WeatherEdit/background_editing/ckpts')
    args = ap.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        cache = output/'.hf_cache'
        provenance = output/'download_sources.json'
        api = HfApi()
        if provenance.exists():
            sources = _hugsim_expand(json.loads(provenance.read_text()))
            sd_revision = sources['sd_turbo_revision']
        else:
            # Resolve before transferring large files; retries retain this revision.
            sd_revision = api.model_info('stabilityai/sd-turbo', timeout=20).sha
            sources = dict(sd_turbo_revision=sd_revision,weather_revision=REVISION,
                           weather_sha256=WEATHER_SHA256,formal_evaluation_admitted=False)
            with provenance.open('x') as f: json.dump(sources,f,indent=2)
        # Existing valid weather weights are reused; other existing files are preserved.
        target = output/'with_seg.pkl'
        if target.exists():
            if digest(target) != WEATHER_SHA256:
                raise RuntimeError('existing with_seg.pkl hash mismatch; preserving it')
        else:
            print('Downloading official with_seg.pkl (2.24 GB), cached/resumable', flush=True)
            path = Path(hf_hub_download('jumponthemoon/WeatherEdit','with_seg.pkl',
                        revision=REVISION,cache_dir=str(cache)))
            if digest(path) != WEATHER_SHA256: raise RuntimeError('weather SHA256 mismatch')
            publish_link(target,path)
        print('Weather SHA256 verified',flush=True)
        info = api.model_info('stabilityai/sd-turbo', revision=sd_revision, timeout=20)
        names = {f.rfilename for f in info.siblings}
        required = ['model_index.json','text_encoder/config.json','unet/config.json','vae/config.json']
        for folder, candidates in (
            ('text_encoder',('model.safetensors','pytorch_model.bin')),
            ('unet',('diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin')),
            ('vae',('diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin'))):
            selected = next((folder+'/'+n for n in candidates if folder+'/'+n in names),None)
            if selected is None: raise RuntimeError('no standard precision weights: '+folder)
            required.append(selected)
        required += sorted(n for n in names if n.startswith(('tokenizer/','scheduler/')))
        if any(n not in names for n in required): raise RuntimeError('missing base model config')
        print('Downloading pinned SD-Turbo components, omitting duplicate fp16/root weights',flush=True)
        snapshot = Path(snapshot_download('stabilityai/sd-turbo',revision=sd_revision,
                        cache_dir=str(cache),allow_patterns=required,max_workers=2))
        for name in required:
            if not (snapshot/name).is_file(): raise RuntimeError('missing downloaded file: '+name)
        publish_link(output/'sd-turbo',snapshot)
        # SHA verified before loading the official pickle. This is a CPU structure check.
        import torch
        data = torch.load(target,map_location='cpu',weights_only=True)
        keys = ['rank_unet','l_target_modules_encoder','l_target_modules_decoder',
                'l_modules_others','rank_vae','vae_lora_target_modules','sd_vae_enc','sd_vae_dec']
        keys += [f'sd_{w}_{p}' for w in ('rainy','snowy','foggy') for p in ('encoder','decoder','other')]
        missing = [k for k in keys if k not in data]
        result = dict(downloaded=True,weather_sha256_verified=True,missing_keys=missing,
                      sd_turbo_revision=sd_revision,sd_turbo_files=required,
                      resource_structure_ready=not missing,gpu_inference_verified=False,
                      formal_evaluation_admitted=False)
        (output/'download_verification.json').write_text(json.dumps(result,indent=2))
        if missing: raise RuntimeError('weather checkpoint schema mismatch: '+str(missing))
        print('Download + CPU structure verification complete. GPU inference remains pending.',flush=True)

if __name__ == '__main__': main()
