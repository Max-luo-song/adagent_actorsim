
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

import json,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;P28=HERE.parent;ROOT=P28.parents[5]
sys.path[:0]=[str(HERE),str(P28),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from screening import retained_reference_check
from heading_fix_v1.trace_near_ego import guard
from heading_fix_v1 import closed172_common as cc

def assess(output):
    output=Path(output);vehicle=cc.load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
    summary=dict(formal_evaluation_admitted=False,closed_loop_success_verified=False,visible_retained_reference_observations=0,visible_reference_changed_or_removed=0,cases={},external_driver_memory_replayed=False)
    for mode in ('baseline','admission'):
        conflicts=[]
        for i in range(81):
            d=output/mode/f'step_{i:03d}'
            with (d/'observation.pkl').open('rb') as f:_,info=pickle.load(f)
            report=_hugsim_expand(json.loads((d/'scene_reliability.json').read_text()))
            with np.load(d/'perception.npz') as data:g,initial=guard(info,report,data['occupancy'],vehicle)
            gap=float(g.clearance(initial[None],[0.])[0])
            if gap<0:conflicts.append(i)
            if mode=='admission':
                original=_hugsim_expand(json.loads((output/f'baseline/step_{i:03d}/scene_reliability.json').read_text()))
                check=retained_reference_check(original,report,info)
                summary['visible_retained_reference_observations']+=check['reference_observations'];summary['visible_reference_changed_or_removed']+=len(check['changed_or_removed'])
        summary['cases'][mode]=dict(current_proposal_initial_conflict_frames=conflicts)
    (output/'screening.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
