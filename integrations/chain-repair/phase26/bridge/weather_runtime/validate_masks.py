"""Small GPU gate: recorded pose vs novel pose vs native edited novel pose."""

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
import gc
import hashlib
import json
import sys
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
JOB=HERE.parents[3];ROOT=JOB.parents[3]
sys.path[:0]=[str(ROOT/'integrations/sr-hugsim'),str(BRIDGE/'framework_backend')]
from scene_api import SceneSpec
from mask_backend import MaskFrameworkBackend,GROUPS
from native_edit import translate_native


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(formal_evaluation_admitted=False,diagnostic_uncertified=True,passed=False,
        recorded_masks_used=False,dynamic_groups=list(GROUPS),coverage_threshold=.2,
        precision='Gaussian semantic coverage; not ground truth mesh segmentation',cases={})
    spec=SceneSpec.read(JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json')
    inputs=Path(spec.source['inputs']);start=spec.source['start_timestep']
    origin=np.loadtxt(inputs/'ego_pose'/f'{start:03d}.txt')
    pose=np.linalg.inv(origin)@np.loadtxt(inputs/'ego_pose'/f'{start+80:03d}.txt')
    novel=pose.copy();novel[:3,3]+=pose[:3,0]*.5
    arrays={}
    def save_case(backend,name,at):
        raw=backend.render(at,8.)
        arrays[name]=raw
        info=dict(ego_to_world=at.tolist(),timestamp_s=8.,cameras={})
        for cam,c in raw['cameras'].items():
            rgb=c['rgb'];mask=c['weather_dynamic_mask'];coverage=c['weather_dynamic_coverage']
            Image.fromarray(rgb).save(args.output/f'{name}_{cam}_rgb.png')
            Image.fromarray(mask).save(args.output/f'{name}_{cam}_mask.png')
            np.save(args.output/f'{name}_{cam}_coverage.npy',coverage)
            overlay=rgb.copy();active=mask>0
            overlay[active]=(rgb[active]*.5+np.array([127,0,0])).clip(0,255).astype('uint8')
            Image.fromarray(overlay).save(args.output/f'{name}_{cam}_overlay.png')
            info['cameras'][cam]=dict(mask_pixels=int(active.sum()),K=np.asarray(c['K']).tolist(),
                camera_to_world=np.asarray(c['camera_to_world']).tolist(),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        report['cases'][name]=info
    try:
        b=MaskFrameworkBackend(spec)
        try:save_case(b,'recorded',pose);save_case(b,'novel',novel)
        finally:b.close();del b;gc.collect()
        import torch
        torch.cuda.empty_cache()
        b=MaskFrameworkBackend(spec)
        try:
            edit=_hugsim_expand(json.loads((BRIDGE/'unified/native_pedestrian_edit.json').read_text()))
            before=b.trainer.models[edit['group']].instances_trans.detach().clone()
            report['native_edit']=translate_native(b,edit)
            after=b.trainer.models[edit['group']].instances_trans.detach()
            report['proxy_translation_verified']=bool(torch.allclose(after[:,edit['instance']]-before[:,edit['instance']],
                after.new_tensor(edit['offset_world_m']).expand_as(after[:,edit['instance']])))
            save_case(b,'edited_novel',novel)
        finally:b.close()
        diffs={}
        for label,a,z in [('pose_change','recorded','novel'),('edit_change','novel','edited_novel')]:
            diffs[label]={cam:dict(mask_changed_pixels=int(np.count_nonzero(
                arrays[a]['cameras'][cam]['weather_dynamic_mask']!=arrays[z]['cameras'][cam]['weather_dynamic_mask'])),
                rgb_changed_pixels=int(np.any(arrays[a]['cameras'][cam]['rgb']!=arrays[z]['cameras'][cam]['rgb'],axis=-1).sum()))
                for cam in arrays[a]['cameras']}
        report['differences']=diffs
        # Newly added semantic raster must not perturb the previously measured RGB path.
        previous=BRIDGE/'unified/weather_window_runs/20261101T112744Z_rainy_3x3/inputs'
        rgb_regression={}
        for c,name in enumerate(('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')):
            with Image.open(previous/f'080_{c}_framework.png') as im: reference=np.array(im)
            current=arrays['recorded']['cameras'][name]['rgb']
            delta=np.abs(reference.astype(float)-current.astype(float))
            rgb_regression[name]=dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()))
        report['original_rgb_regression']=rgb_regression
        unchanged=all(x['max_abs']<=3 and x['mean_abs']<=.2 for x in rgb_regression.values())
        report['passed']=unchanged and report['proxy_translation_verified'] and all(any(
            d['mask_changed_pixels']>0 and d['rgb_changed_pixels']>0 for d in cameras.values()) for cameras in diffs.values())
        if not report['passed']:raise RuntimeError('mask GPU gate failed; inspect report')
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:(args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
