"""Saved-observation regression: sensor geometry unchanged, state restorable."""

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

import copy,json,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import global_repair_v2.run_harness
from global_repair_v2.unknown_recovery import RecoveryDepthMemory
RUN=HERE.parent/'runs/global_evidence_closed552_v1'

def main():
    with (RUN/'branch_snapshot/driver_snapshot.pkl').open('rb') as f:old=pickle.load(f)
    memory=old['depth_memory'];memory.__class__=RecoveryDepthMemory;memory.recovery_episode=None
    rows=[];deadline=None
    for index in range(old['next_index'],67):
        folder=RUN/f'uniad_worker/step_{index:03d}'
        with (folder/'observation.pkl').open('rb') as f:obs,info=pickle.load(f)
        points,audit=memory.update(obs,info);saved=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()))
        np.testing.assert_array_equal(points,np.asarray(saved['observed_depth_obstacles_xy']).reshape(-1,2))
        np.testing.assert_array_equal(audit['unknown_visibility_points_xy'],saved['unknown_visibility_obstacles_xy'])
        meta=audit['unknown_visibility_geometry'].pop('initial_unknown_margin_recovery')
        assert _hugsim_expand(json.loads(json.dumps(audit['unknown_visibility_geometry'])))==saved['unknown_visibility_geometry']
        if meta['episode']:
            if deadline is None:deadline=meta['episode']['deadline'];assert index==36
            assert meta['episode']['deadline']==deadline
        if index==38:
            restored=pickle.loads(pickle.dumps(memory,protocol=4))
            assert restored.recovery_episode==memory.recovery_episode;memory=restored
        rows.append(dict(step=index,depth_and_unknown_arrays_identical=True,recovery=meta))
        if index in (21,36,38,50,66):print('Sensor memory replay',index,flush=True)
    result=dict(scope='Fixed saved sensor inputs; no inference, MPC or new control',frames=len(rows),
        all_sensor_geometry_identical=True,deadline_persisted_across_31_replans=True,
        active_recovery_pickle_roundtrip=True,unknown_voxels_removed=0,frames_detail=rows)
    out=HERE.parent/'runs/unknown_recovery_mpc_probe';out.mkdir(exist_ok=True)
    (out/'memory_replay.json').write_text(json.dumps(result,indent=2)+'\n');print('OUTPUT:',out/'memory_replay.json',flush=True)
if __name__=='__main__':main()
