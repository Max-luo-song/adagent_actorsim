"""Shadow-mode evaluation on saved observations; never rewrites detections."""

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
sys.path[:0]=[str(HERE),str(ROOT/'waymo23-native-import'),str(ROOT/'code/UniAD_SIM/projects/mmdet3d_plugin/uniad')]
from lifecycle import QueryLifecycle

def main():
    source=P28/'runs/lineage92_closed_scene552_v4/uniad_worker'
    gate=None;events=[];all_rows=[];protected=0
    for index in range(104):
        d=source/f'step_{index:03d}'
        report=_hugsim_expand(json.loads((d/'scene_reliability.json').read_text()));q=_hugsim_expand(json.loads((d/'neural_query_origin.json').read_text()))
        if gate is None:gate=QueryLifecycle(report['config'])
        with (d/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        with np.load(d/'perception.npz') as a:boxes=a['candidate_boxes'].copy();labels=a['candidate_labels'].copy()
        result=gate.observe(boxes,q['selected_track_ids'],labels,obs,info)
        protected+=sum(bool(r['protected_observation']) for r in result['queries'])
        all_rows.append(dict(step=index,**result))
        if result['retire_next_forward']:events.append(dict(step=index,ids=result['retire_next_forward']))
        if index%10==0:print('Shadow replay',index,flush=True)
    output=P28/'runs/proposal_evidence_shadow_v1';output.mkdir(exist_ok=False)
    (output/'history.json').write_text(json.dumps(all_rows,indent=2))
    (output/'summary.json').write_text(json.dumps(dict(mode='shadow_saved_outputs',retirement_suggestions=events,protected_query_observations=protected,
        all_retirement_ids=sorted({i for e in events for i in e['ids']}),raw_arrays_modified=False,new_inference_executed=False,closed_loop_success_verified=False),indent=2))
    print('Shadow summary:',output/'summary.json',flush=True)
if __name__=='__main__':main()
