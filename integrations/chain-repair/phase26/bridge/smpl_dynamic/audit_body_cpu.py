"""Checkpoint-only body-motion audit. Does NOT replace GPU rendering acceptance."""

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

import json
import sys
pass  # Path is provided by the portable boundary above
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from deform_export import deform_forward
from shape_metrics import shape_residual
job=HERE.parents[3]
manifest=_hugsim_expand(json.loads((job/'chain_repair/phase4/dynamic552_frame80/scene_manifest.json').read_text()))
source=_hugsim_expand(json.loads(Path(manifest['source_report']).read_text()))
checkpoint=Path(source['checkpoint'])
models=torch.load(checkpoint,map_location='cpu',weights_only=False)['models']
smpl=models['SMPLNodes']; deform=models['DeformableNodes']
q=smpl['smpl_qauts'].numpy().copy()
q/=np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-12)
angles=2*np.arccos(np.clip(np.abs((q[80]*q[85]).sum(-1)),0,1))
mask=(deform['points_ids'].reshape(-1)==1)
x=deform['_means'][mask].double()
height=deform['instances_size'][1,2]
embed=deform['instances_embedding'][1].double().reshape(1,-1)
with torch.no_grad():
    a=deform_forward(deform,x/height*2,80/(deform['instances_fv'].shape[0]-1),embed)[0]
    b=deform_forward(deform,x/height*2,85/(deform['instances_fv'].shape[0]-1),embed)[0]
result=dict(diagnostic_uncertified=True,formal_evaluation_admitted=False,
    evidence_level='CPU checkpoint parameters and standalone mirrored network only',
    gpu_render_acceptance=False,
    checkpoint=str(checkpoint),checkpoint_size=checkpoint.stat().st_size,
    smpl=dict(body_joint_max_angle_rad=float(angles.max()),
              instances_with_joint_change_gt001rad=int((angles.max(-1)>.01).sum())),
    deform_instance1=dict(shape_residual=shape_residual((x+a).numpy(),(x+b).numpy()),
                          time_delta_rms_m=float(torch.sqrt(((b-a)**2).sum(-1).mean()))))
out=HERE/'cpu_body_audit_20261101.json'
with out.open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
