"""GPU image comparison for the new SR material implementation.

Uses the existing HUGSIM geometry/rasterizer for both arms and replaces only
lighting + PBR with actual SR source implementations. This verifies the material
bridge; it deliberately does not claim full MultiTrainer/online editing validation.
"""

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
import inspect
import json
import os
pass  # Path is provided by the portable boundary above
import sys
import subprocess
from types import SimpleNamespace
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE/'waymo23-native-import'))
from pbr_validation import validate, background_from_state, pbr, Config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the image comparison')
    source_path=Path(args.source_report).resolve()
    source=_hugsim_expand(json.loads(source_path.read_text()))
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    # A separate process avoids mixing SR's models/utils packages with HUGSIM's.
    subprocess.run([sys.executable, str(BASE/'integrations/sr-hugsim/production_material_load.py'),
                    '--output', str(output/'production_material_load.json')],
                   cwd=BASE, check=True, timeout=600)
    report=validate(source_path,device='cuda')
    report['production_material_load_verified'] = True
    torch.cuda.empty_cache()
    os.environ['BASE']=str(BASE)
    sys.path[:0]=[str(BASE/'code/HUGSIM'), str(BASE/'code/HUGSIM/sim')]
    import waymo552_three
    from gsplat import spherical_harmonics
    from PIL import Image
    env=None
    scope=None
    originals={}
    try:
        env=waymo552_three.make_env(scale=.25,source_report=str(source_path))
        ctrl=Config(env.cfg['trainer']['gaussian_ctrl_general_cfg'] | env.cfg['model']['Background']['ctrl'])
        background=background_from_state(env.models['Background'],ctrl,'cuda',spherical_harmonics)
        state=env.models['GlobalLighting']
        n=state['appearance_embedding.weight'].shape[0]
        light=pbr.GlobalLightingModule(class_name='GlobalLighting',n=n,device=torch.device('cuda'),
            **env.cfg['model']['GlobalLighting']['params']).cuda()
        light.load_state_dict(state,strict=True)
        light.eval().requires_grad_(False)
        scope=inspect.unwrap(env.observe.__func__).__globals__
        originals={name:scope[name] for name in ('pbr_color','lighting')}
        def color_bridge(received,position,coeff):
            if received is not env.models['Background']:
                raise ValueError('Unexpected material source')
            cam=torch.eye(4,device=position.device);cam[:3,3]=position
            return background.compute_pbr_color(SimpleNamespace(camtoworlds=cam),coeff)
        def light_bridge(received,frame,frames,image_id):
            if received is not env.models['GlobalLighting']:
                raise ValueError('Unexpected light source')
            return pbr.lighting_for_frame(light,frame,image_id,n,'cuda')
        comparisons=[]
        for frame in (0,source['frames']//2,source['frames']-1):
            env.motion.timestamp=frame/env.fps
            scope.update(originals)
            before=env.observe()
            scope.update(pbr_color=color_bridge,lighting=light_bridge)
            after=env.observe()
            for camera in before['rgb']:
                rgb_error=np.abs(before['rgb'][camera].astype(np.int16)-after['rgb'][camera].astype(np.int16))
                depth_error=np.abs(before['depth'][camera]-after['depth'][camera])
                if rgb_error.max()>1 or rgb_error.mean()>.1 or not np.allclose(before['depth'][camera],after['depth'][camera],atol=1e-6,rtol=1e-6):
                    raise RuntimeError(f'Material bridge image mismatch at {frame}/{camera}')
                Image.fromarray(np.concatenate([before['rgb'][camera],after['rgb'][camera]],axis=1)).save(output/f'{frame:03d}_{camera}_legacy_vs_sr_pbr.png')
                comparisons.append(dict(frame=frame,camera=camera,rgb_max_uint8_error=int(rgb_error.max()),
                    rgb_mean_uint8_error=float(rgb_error.mean()),depth_max_abs_error=float(depth_error.max())))
        report.update(status='material_bridge_gpu_comparison_passed',image_comparisons=comparisons,
            material_bridge_image_verified=True, full_renderer_verified=False,
            full_trainer_imported=False, integration_complete=False,
            geometry_backend='existing_waymo_adapter',rasterizer='existing_hugsim',
            ego_pose_mode='fixed_initial_ego_pose_for_paired_material_comparison',
            limitations=['Not a full SR MultiTrainer render','No new closed-loop or online-edit validation'])
        (output/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if scope is not None:
            scope.update(originals)
        if env is not None:
            env.close()


if __name__=='__main__':
    main()
