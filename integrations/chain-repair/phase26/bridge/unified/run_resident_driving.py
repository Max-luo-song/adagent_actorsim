"""Use the normal weather driving path across resets/scenes with ONE UniAD."""

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
pass  # Path is provided by the portable boundary above
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
ROOT_PATH=next(p for p in HERE.parents if (p/'code/UniAD_SIM').is_dir())
sys.path.insert(0,str(BRIDGE/'weather_runtime'))
import run_weather_loop
from resident_inference import ResidentInferenceService,atomic_json
from shared_weather_scene import SharedWeatherScene
from run_resident_validation import preflight
from reuse_articulation import verify
from checkpoint_preflight import verify as verify_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.steps<=117:parser.error('--steps must be 1..117')
    primary=BRIDGE.parent.parent/'phase4/dynamic552_frame80/scene_manifest.json'
    secondary=HERE/'scene_switch_551.json'
    a,b=preflight(primary,secondary)
    body=HERE/'edited_weather_runs/20261102T031621Z'
    reuse=verify(body)
    checkpoint_probe=verify_checkpoint()
    if args.check_only:
        print(json.dumps(dict(passed=True,scenes=[a.source['scene'],b.source['scene']],
            controls_per_episode=args.steps,body_source_checks=len(reuse['source_checks']),
            checkpoint_preflight_passed=checkpoint_probe['passed'],gpu_executed=False)))
        return
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source_paths=list(HERE.glob('*.py'))+list((BRIDGE/'weather_runtime').glob('*.py'))
    source_paths += [BRIDGE.parent.parent/'phase11/paired_closed_loop_arm.py',
        BRIDGE/'fast_control/driver_state_audit.py',BRIDGE/'fast_control/tracking_common.py',
        BRIDGE/'fast_control/planning_guard_tracking.py',BRIDGE/'fast_control/checkpoint_preflight.py',
        BRIDGE/'framework_backend/backend.py',BRIDGE/'framework_backend/native_edit.py',
        ROOT_PATH/'waymo23-native-import/resident_session_state.py',
        ROOT_PATH/'waymo23-native-import/waymo172_closed_loop_v1/closed172_worker.py',HERE/'run_phase26.sh']
    atomic_json(out/'implementation.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    atomic_json(out/'body_reuse.json',reuse)
    atomic_json(out/'checkpoint_preflight.json',checkpoint_probe)
    plan=[('scene552_first',primary,True),('scene552_reset',primary,True),
          ('scene551_switch',secondary,False),('scene552_return',primary,True)]
    completed=[]
    try:
        with ResidentInferenceService(out/'inference_service') as service,SharedWeatherScene(out/'scene_host') as host:
            for name,manifest,edited in plan:
                print(f'Starting real resident driving episode: {name}',flush=True)
                options=argparse.Namespace(output=out/name,steps=args.steps,native_edit=None,
                    edit_command=HERE/'native_pedestrian_command.json' if edited else None,
                    articulation_report=body/'articulation/report.json' if edited else None,
                    verify_edit_lifecycle=True)
                acceptance=run_weather_loop.run_episode(options,inference_service=service,
                    scene_host=host,scene_manifest=manifest)
                report=_hugsim_expand(json.loads((out/name/'closed_loop/run_report.json').read_text()))
                completed.append(dict(name=name,acceptance=acceptance,
                    scene=report['scene'],steps=len(report['trace']),worker_ready=report['worker_ready'],
                    driver_state_initial=report['driver_state_initial']))
                atomic_json(out/'completed_episodes.json',completed)
                print(f'{name}: {len(report["trace"])} executed feedback controls completed',flush=True)
            checks=dict(four_driving_episodes=len(completed)==4,
                all_episode_acceptance=all(e['acceptance']['passed'] for e in completed),
                control_counts=all(e['steps']==args.steps for e in completed),
                scene_sequence=[e['scene'] for e in completed]==[552,552,551,552],
                one_model=all(e['worker_ready']['resident_identity']==service.ready['resident_identity'] for e in completed),
                all_control_state_fresh=all(all(e['driver_state_initial']['checks'].values()) for e in completed),
                first_two_reuse_scene=host.transitions[1]['same_backend_reset'] and
                    host.transitions[0]['backend_object_id']==host.transitions[1]['backend_object_id'],
                real_scene_switch_and_return=all(t['scene_switched'] and t['old_backend_closed'] for t in host.transitions[2:]),
                loaded_scene_models=bool(host.transitions[0]['gaussian_points_by_group']) and
                    host.transitions[0]['gaussian_points_by_group']!=host.transitions[2]['gaussian_points_by_group'] and
                    host.transitions[0]['gaussian_points_by_group']==host.transitions[3]['gaussian_points_by_group'],
                same_scene_session=len({t['session_object_id'] for t in host.transitions})==1,
                model_served_all_controls=len(service.records)==len(plan)*args.steps,
                inference_episode_isolation=all(r['reset']==(r['local_index']==0) and r['history']==(r['local_index']>0) for r in service.records))
            receipts=[_hugsim_expand(json.loads((out/name/'weather_input_receipts.json').read_text())) for name,_,_ in plan[:2]]
            checks['reset_same_inputs_reproduced']=receipts[0]==receipts[1]
            reports=[_hugsim_expand(json.loads((out/name/'closed_loop/run_report.json').read_text())) for name,_,_ in plan[:2]]
            control_diff=max(float(np.max(np.abs(np.asarray(x['state_after'])-np.asarray(y['state_after']))))
                for x,y in zip(reports[0]['trace'],reports[1]['trace']))
            checks['reset_same_control_states']=np.isfinite(control_diff) and control_diff<=1e-6
            atomic_json(out/'acceptance.json',dict(passed=all(checks.values()),checks=checks,
                same_scene_max_state_difference=control_diff,model_loads=1,resident_pid=service.ready['pid'],
                closed_loop_driving=True,control='iLQR tracking passthrough',safety_certified=False,
                formal_evaluation_admitted=False))
            if not all(checks.values()):raise RuntimeError('Resident driving acceptance failed; inspect acceptance.json')
    except BaseException as exc:
        atomic_json(out/'failure.json',dict(error=repr(exc),completed_episodes=[e['name'] for e in completed],formal_evaluation_admitted=False))
        raise


if __name__=='__main__':main()
