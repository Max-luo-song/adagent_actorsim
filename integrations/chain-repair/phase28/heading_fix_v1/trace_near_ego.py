"""Offline attribution only: never changes detector output or runtime policy."""

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

import copy
import hashlib
import json
import pickle
pass  # Path is provided by the portable boundary above
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
P28 = HERE.parent
ROOT = P28.parents[5]
sys.path.insert(0, str(P28))
from heading_fix_v1 import closed172_common as cc

RUN = P28/'runs/heading_fix_scene552_v1'
OUT = P28/'runs/near_ego_causal_audit_20261102'

def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else float(x)))

def load(step):
    p=RUN/f'uniad_worker/step_{step:03d}'
    with (p/'sensor_metadata.pkl').open('rb') as f: info=pickle.load(f)
    rel=_hugsim_expand(json.loads((p/'guard_memory_report.json').read_text()))
    with np.load(p/'perception.npz') as d:
        data={k:d[k].copy() for k in d.files}
    return p,info,rel,data

def guard(info,rel,masks,vehicle):
    offset=np.asarray(info['model_lidar_frame']['virtual_to_ego'])[:2,3]
    state=np.array([*-offset,0.,info['ego_velo'],info['ego_steer']])
    states=[state.copy()]
    reaction=rel['config']['confirmation_seconds']
    for tick in range(60):
        a=0. if tick*vehicle.dt<reaction-1e-9 else -min(vehicle.emergency_braking_mps2,states[-1][3]/vehicle.dt)
        states.append(cc.integrate_state(states[-1],[a,0.],vehicle.dt,vehicle.wheelbase_m))
    masks=np.asarray(masks).reshape(1,5,200,200)
    return cc.PlanningGuard(rel,vehicle,np.asarray(states),np.arange(61)*vehicle.dt,current_occupancy=masks[0,0]),state

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'chronology.json').exists(), 'Refusing existing results'
    vehicle=cc.load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json')
    hashes={}
    chronology=[]
    counterfactuals={}
    # Terminal observation69 lacks the original sensor_metadata.pkl; do not
    # silently substitute metadata from a different frame.
    for step in range(69):
        p,info,rel,data=load(step)
        for f in ('sensor_metadata.pkl','guard_memory_report.json','perception.npz','optimization.json'):
            hashes[str(p/f)]=hashlib.sha256((p/f).read_bytes()).hexdigest()
        g,initial=guard(info,rel,data['occupancy'],vehicle)
        blockers=[]
        for o, audit in zip(rel['obstacles'],g.rows):
            # A nonnegative full braking-envelope bound cannot be an
            # initial-state blocker; only isolate possible conflicting rows.
            if not audit['hard_constraint'] or audit['braking_envelope_clearance_m']>=0:
                continue
            one=copy.deepcopy(rel);one['obstacles']=[o]
            single,_=guard(info,one,data['occupancy'],vehicle)
            gap=float(single.clearance(initial[None],np.zeros(1))[0])
            if gap < -1e-5:
                blockers.append(dict(candidate_index=o['candidate_index'],memory_id=o.get('memory_id'),
                    evidence_track_id=o.get('evidence_track_id'),state=o['state'],score=o['raw_score'],
                    observations=o.get('observations'),duration=o.get('duration_seconds'),
                    missing_seconds=o.get('missing_seconds'),clearance=gap,box=o['box']))
        offset=-initial[:2]
        boxes=data['candidate_boxes'];scores=data['candidate_scores'];labels=data['candidate_labels']
        nearby=[]
        for i in range(len(boxes)):
            distance=float(np.linalg.norm(boxes[i,:2]+offset))
            if int(labels[i])==0 and distance < 3. and scores[i]>=.05:
                nearby.append(dict(index=i,score=float(scores[i]),distance_to_ego=distance,box=boxes[i].tolist()))
        opt=_hugsim_expand(json.loads((p/'optimization.json').read_text()))
        chronology.append(dict(step=step,timestamp=info.get('timestamp'),speed=info['ego_velo'],
            reason=opt.get('reason'),initial_conflict=opt.get('initial_conflict'),
            blockers=blockers,nearby_raw_candidates=nearby))
        print('audited',step,'blockers',[o['candidate_index'] for o in blockers],flush=True)
        if step in (39,46,60):
            remove={o['candidate_index'] for o in blockers}
            reduced=copy.deepcopy(rel);reduced['obstacles']=[o for o in rel['obstacles'] if o['candidate_index'] not in remove]
            g2,_=guard(info,reduced,data['occupancy'],vehicle)
            counterfactuals[str(step)]=dict(removed_indices=sorted(remove),
                original_guard_clearance=float(g.clearance(initial[None],[0.])[0]),
                without_blockers_guard_clearance=float(g2.clearance(initial[None],[0.])[0]),
                remaining_obstacles=len(reduced['obstacles']))
    dump('chronology.json',chronology)
    dump('initial_counterfactuals.json',counterfactuals)
    dump('input_sha256.json',hashes)
    print('chronology and pointwise ablations saved',flush=True)
    step=39
    p,info,rel,data=load(step)
    original=_hugsim_expand(json.loads((p/'optimization.json').read_text()))
    original_result=cc.optimize(data['trajectory'],data['occupancy'],info,reliability=rel,vehicle=vehicle)
    dump('step39_original_recomputed.json',original_result)
    reduced=copy.deepcopy(rel)
    reduced['obstacles']=[o for o in rel['obstacles'] if o['candidate_index']!=33]
    prior=_hugsim_expand(json.loads((RUN/'uniad_worker/step_038/optimization.json').read_text()))
    warm=np.asarray(prior['selected']['controls']) if prior.get('selected') else None
    started=time.perf_counter()
    result=cc.optimize(data['trajectory'],data['occupancy'],info,warm=warm,reliability=reduced,vehicle=vehicle,_reuse_verified_warm=True)
    dump('step39_without_candidate33_optimization.json',result)
    selected=result.get('selected')
    dump('summary.json',dict(diagnostic_uncertified=True,formal_evaluation_admitted=False,
        original_reason=original_result.get('reason'),recorded_reason=original.get('reason'),
        counterfactual_reason=result.get('reason'),counterfactual_feasible=bool(selected and selected['feasible']),
        selected=selected,wall_seconds=time.perf_counter()-started,
        scope='Offline removal of candidate33 only, frozen sensing/occupancy/road/vehicle/heading-corrected optimizer. Not a deployable gate, new observation, closed-loop success or certified absence.'))
    for name,digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    print('completed',flush=True)

if __name__=='__main__':main()
