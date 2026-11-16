"""Isolate mask hook effects from cross-run RGB reproducibility; one model load."""

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
import hashlib
import json
import random
import sys
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;BRIDGE=HERE.parent;JOB=HERE.parents[3];ROOT=JOB.parents[3]
sys.path[:0]=[str(ROOT/'integrations/sr-hugsim'),str(BRIDGE/'framework_backend')]
from scene_api import SceneSpec
from backend import FrameworkSceneBackend
from mask_backend import MaskFrameworkBackend


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(formal_evaluation_admitted=False,diagnostic_uncertified=True,
                diagnostic_complete=False,mask_gate_admitted=False,comparisons={})
    backend=None
    try:
        import torch
        spec=SceneSpec.read(JOB/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json')
        inp=Path(spec.source['inputs']);start=spec.source['start_timestep']
        pose=np.linalg.inv(np.loadtxt(inp/'ego_pose'/f'{start:03d}.txt'))@np.loadtxt(inp/'ego_pose'/f'{start+80:03d}.txt')
        # Seed all initializations; compare modes on the same loaded parameters.
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        backend=MaskFrameworkBackend(spec)
        wrapped=backend.trainer.render_gaussians
        original=backend._plain_raster
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        renders={}
        for label,enabled in (('plain_before',False),('masked',True),('plain_after',False)):
            random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            backend.trainer.render_gaussians=wrapped if enabled else original
            raw=backend.render(pose,8.) if enabled else FrameworkSceneBackend.render(backend,pose,8.)
            renders[label]=raw
            for cam,c in raw['cameras'].items():Image.fromarray(c['rgb']).save(args.output/f'{label}_{cam}.png')
        previous=BRIDGE/'unified/weather_window_runs/20261101T112744Z_rainy_3x3/inputs'
        for c,cam in enumerate(('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')):
            a=renders['plain_before']['cameras'][cam]['rgb']
            references={label:renders[label]['cameras'][cam]['rgb'] for label in ('masked','plain_after')}
            with Image.open(previous/f'080_{c}_framework.png') as im:references['historical']=np.array(im)
            report['comparisons'][cam]={}
            for label,b in references.items():
                d=np.abs(a.astype(float)-b.astype(float))
                report['comparisons'][cam][label]=dict(max_abs=float(d.max()),mean_abs=float(d.mean()))
                Image.fromarray(np.clip(d*8,0,255).astype('uint8')).save(args.output/f'diff_plain_vs_{label}_{cam}.png')
        within=lambda key:all(d[key]['max_abs']<=3 and d[key]['mean_abs']<=.2 for d in report['comparisons'].values())
        report.update(diagnostic_complete=True,controlled_mask_rgb_unchanged=within('masked'),
            controlled_repeat_stable=within('plain_after'),historical_plain_reproduced=within('historical'),
            seed=42,pose=pose.tolist(),timestamp_s=8.,thresholds=dict(max_abs=3,mean_abs=.2),
            source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (
                Path(__file__).resolve(),HERE/'mask_backend.py',BRIDGE/'framework_backend/backend.py')})
        if within('masked') and within('plain_after') and not within('historical'):
            report['finding']='historical drift reproduced WITHOUT mask hook; exact cause remains unresolved'
        else:report['finding']='inspect controlled comparisons; do not infer cause from historical comparison alone'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if backend is not None:backend.close()
        (args.output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
