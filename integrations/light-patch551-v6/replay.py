"""Replay actual recorded observations; exports hypothetical references, never ego motion."""

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

import argparse,json,pickle,sys
pass  # Path is provided by the portable boundary above
import numpy as np
LOCAL=Path(__file__).resolve().parent;ROOT=LOCAL.parents[1]
sys.path[:0]=[str(LOCAL),str(LOCAL.parent/'light-map551-v4'),str(LOCAL.parent/'light-cutin-diagnostic')]
from common import P26,asset,read,write,configure
from front_brake import apply,PARAMETERS
OUTPUT=P26/'light_patch551_v6'
OFF={'A':P26/'light_map551_v4/A','B':P26/'light_map551_cutin_v5/B'}

def ego_inputs(info,motion,vehicle):
    return dict(dt_s=vehicle.dt,speed_mps=float(motion.speed),steering_rad=float(motion.steer),
        half_length_m=vehicle.half_length_m,half_width_m=vehicle.half_width_m,min_acc_mps2=vehicle.min_acc_mps2,
        virtual_to_ego=info['model_lidar_frame']['virtual_to_ego'],
        lidar_to_world_rotation=np.asarray(info['explicit_l2g_r_mat']).tolist(),
        lidar_to_world_translation=np.asarray(info['explicit_l2g_t']).tolist())

def run(root=OUTPUT):
    from types import SimpleNamespace
    configure()
    from runtime_configuration import load_vehicle_config
    from map_reference import MapRoadReference
    vehicle=load_vehicle_config();root=Path(root);summaries={}
    for name,folder in OFF.items():
        out=root/'replay'/name;out.mkdir(parents=True,exist_ok=True)
        source=read(folder/'run_report.json');policy=MapRoadReference(vehicle,read(P26/'light_map551_v4/map/map.json'))
        state=None;rows=[];changed=[];triggered=[];certificates=[]
        for row in source['trace']:
            i=row['index'];step=folder/f'uniad_worker/step_{i:03d}'
            with (step/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
            with np.load(step/'perception.npz') as z:perception={k:z[k].copy() for k in ['boxes','scores','labels']}
            base=np.asarray(row['adopted']);motion=SimpleNamespace(speed=row['motion_state'][3],steer=row['motion_state'][4])
            output=apply(base,perception,ego_inputs(info,motion,vehicle),state);state=output['state'];reference=output['reference'];audit=output['audit']
            policy.mount=np.asarray(info['model_lidar_frame']['virtual_to_ego']);policy.geometry.set_pose(info)
            cert=policy.certificate(policy.states(reference));certificates.append(cert['passed'])
            if audit['reference_changed']:changed.append(i)
            if audit['triggered']:triggered.append(i)
            record=dict(index=i,baseline_adopted=base.tolist(),patched_reference=reference.tolist(),patch=audit,map_certificate=cert,
                original_observation=asset(step/'perception.npz'),current_metadata=asset(step/'sensor_metadata.pkl'),
                actual_control_executed=False,offline_fixed_observations_only=True)
            write(out/f'references/{i:03d}.json',record);rows.append(record)
        result=dict(arm=name,original_actual_run=asset(folder/'run_report.json'),parameters=PARAMETERS,
            observed_control_origins=len(rows),changed_reference_count=len(changed),changed_indices=changed,
            triggered_count=len(triggered),all_reference_map_certificates=all(certificates),
            first_changed_index=changed[0] if changed else None,last_changed_index=changed[-1] if changed else None,
            source_fixed_observations=True,new_closed_loop_executed=False,native_scores_computed=False,
            deployment_admitted=False,candidate_status='draft_pending_actual_B_repeat_and_GPU_validation')
        write(out/'report.json',result);summaries[name]=result
    write(root/'replay/summary.json',summaries);return summaries

if __name__=='__main__':print(json.dumps(run(),indent=2))
