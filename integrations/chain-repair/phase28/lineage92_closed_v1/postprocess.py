"""Actual trajectory/pass/return evidence, robust to terminal metadata gaps."""

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

import hashlib
import json
import os
pass  # Path is provided by the portable boundary above
import subprocess
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]
sys.path[:0]=[str(P28),str(P28.parent/'phase11'),str(ROOT/'integrations/sr-hugsim'),str(ROOT/'waymo23-native-import')]
from heading_fix_v1.closed172_common import asset_reference_vertices,body_vertices,load_vehicle_config
from heading_fix_v1.postprocess_recover_v2 import recovered_metadata
from waymo_adapter import SR_TO_HUGSIM as T

def outcome(report,asset,vehicle):
    observations=report.get('observations',[])
    if not observations:return dict(observations=0,passed=False,returned=False)
    first=T@np.asarray(observations[0]['closed_ego_to_world'])@T.T
    forward=first[[0,2],2];forward=forward/np.linalg.norm(forward)
    far=float(np.max(asset_reference_vertices(asset)[:,[0,2]]@forward))
    trace={r['step']:r for r in report.get('trace',[])}
    clear_since=return_since=None;passed=returned=False;rows=[]
    for o in observations:
        pose=T@np.asarray(o['closed_ego_to_world'])@T.T
        direction=pose[[0,2],2];direction=direction/np.linalg.norm(direction)
        yaw=float(np.degrees(np.arctan2(forward[1]*direction[0]-forward[0]*direction[1],forward@direction)))
        rear=float(np.min(body_vertices(pose,vehicle)[:,[0,2]]@forward))
        tick=o['index'];row=trace.get(tick)
        if row is not None:steer=float(row['state_before'][4])
        elif tick-1 in trace:steer=float(trace[tick-1]['state_after'][4])
        else:steer=None
        now=o['timestamp'];clear=rear>far
        clear_since=(now if clear_since is None else clear_since) if clear else None
        passed=passed or (clear_since is not None and now-clear_since>=.5-1e-8)
        straight=clear and abs(yaw)<=5 and steer is not None and abs(np.degrees(steer))<=5 and o['closed_speed_mps']>=.1
        return_since=(now if return_since is None else return_since) if straight else None
        returned=returned or (return_since is not None and now-return_since>=.5-1e-8)
        rows.append(dict(step=tick,timestamp=now,rear_past_asset_m=rear-far,heading_error_deg=yaw,
            steering_deg=None if steer is None else float(np.degrees(steer)),speed=o['closed_speed_mps']))
    return dict(observations=len(observations),passed=bool(passed),returned=bool(returned),last=rows[-1],rows=rows,
        criterion='Full rear beyond asset for >=0.5s; return: full pass + heading<=5deg + steering<=5deg + speed>=0.1m/s for >=0.5s. Diagnostic only.')

def main():
    run=Path(sys.argv[1]).resolve();out=run/'analysis';out.mkdir(exist_ok=False)
    report_path=run/'run_report.json';digest=hashlib.sha256(report_path.read_bytes()).hexdigest()
    report=_hugsim_expand(json.loads(report_path.read_text()));asset=_hugsim_expand(json.loads((run/'asset_reference.json').read_text()))
    result=outcome(report,asset,load_vehicle_config(ROOT/'waymo23-native-import/configs/vehicle.json'))
    view=out/'source_view';view.mkdir();(view/'uniad_worker').mkdir();recovery=[]
    for name in ['run_report.json','scene_manifest.json','insert_request.json','arm_request.json']:(view/name).symlink_to(run/name)
    for step in sorted((run/'uniad_worker').glob('step_[0-9][0-9][0-9]')):
        target=view/'uniad_worker'/step.name
        if (step/'sensor_metadata.json').exists() or not (step/'optimization.json').exists():target.symlink_to(step);continue
        target.mkdir()
        for f in step.iterdir():(target/f.name).symlink_to(f)
        recovery.append(recovered_metadata(step,target/'sensor_metadata.json'))
    (out/'metadata_recovery.json').write_text(json.dumps(recovery,indent=2))
    errors=[]
    for script,args in [('native_eval_export.py',['--run',str(view),'--out',str(out/'native')]),
                        ('native_eval_clouds.py',['--run',str(view),'--out',str(out/'native')]),
                        ('native_eval_score.py',['--export',str(out/'native')])]:
        try:subprocess.run([sys.executable,str(P28/script),*args],check=True)
        except Exception as e:errors.append(dict(stage=script,error=repr(e)));break
    try:
        from comparison_video import encode
        if report.get('observations'):encode(run,10,len(report['observations']),filename='lineage92_closed_3CAM.mp4')
    except Exception as e:errors.append(dict(stage='video',error=repr(e)))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/lineage92-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for label,path in [('original heading fix',P28/'runs/heading_fix_scene552_v1'),('remove neural track92',run)]:
        r=_hugsim_expand(json.loads((path/'run_report.json').read_text()));obs=r.get('observations',[])
        if not obs:continue
        xy=np.array([np.asarray(o['closed_ego_to_world'])[:2,3] for o in obs]);times=np.array([o['timestamp']-obs[0]['timestamp'] for o in obs])
        axes[0].plot(xy[:,0],xy[:,1],label=label)
        axes[1].plot(times,[o['closed_speed_mps'] for o in obs],label=label)
        axes[2].plot([r['timestamp']-obs[0]['timestamp'] for r in r['trace']],[np.degrees(r['state_before'][4]) for r in r['trace']],label=label)
    axes[0].set_aspect('equal');axes[0].set_title('Executed source-world trajectory')
    axes[1].set_title('Executed speed');axes[2].set_title('Front-wheel angle')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.suptitle('Single-case post-hoc lineage intervention; longer window; not general fix validation')
    fig.tight_layout();fig.savefig(out/'trajectory_overlay.png',dpi=160);plt.close(fig)
    removed=[]
    for f in (run/'uniad_worker').glob('step_*/lineage92_filter.json'):
        d=_hugsim_expand(json.loads(f.read_text()))
        if d['removed']:removed.append(dict(step=f.parent.name,removed=d['removed']))
    safety_failure=report['status']=='failed' or 'overlap' in str(report.get('error','')).lower()
    result.update(simulation_status=report['status'],termination_reason=report.get('termination_reason'),
        simulation_error=report.get('error'),controls=len(report.get('trace',[])),wall_seconds=report['wall_seconds'],
        safety_failure=safety_failure,
        observed_pass_and_return=bool(result.get('passed') and result.get('returned') and not safety_failure),
        removal_events=removed,postprocess_errors=errors,formal_evaluation_admitted=False,generalizable_fix=False)
    score=out/'native/native_score.json'
    if score.exists():result['native_score']=_hugsim_expand(json.loads(score.read_text()))
    assert hashlib.sha256(report_path.read_bytes()).hexdigest()==digest
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','removal_events')},indent=2),flush=True)

if __name__=='__main__':main()
