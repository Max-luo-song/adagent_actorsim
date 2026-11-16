"""Recover presentation for a safety-stopped run; original artifacts stay intact."""

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
import pickle
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]
RUN=P28/'runs/heading_fix_scene552_v1'
OUT=RUN/'analysis/recovery_v2'


def recovered_metadata(step, destination):
    """Recover only the exporter's required mount from the actual saved input."""
    original=step/'sensor_metadata.json'
    if original.exists():
        destination.symlink_to(original)
        return None
    packet=step/'observation.pkl'
    if not packet.exists():
        raise ValueError('No original metadata or saved input: '+str(step))
    with packet.open('rb') as fh:
        _,info=pickle.load(fh)
    mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'],dtype=float)
    if mount.shape!=(4,4) or not np.isfinite(mount).all():
        raise ValueError('Invalid saved mount at '+str(step))
    destination.write_text(json.dumps(dict(model_lidar_frame=dict(virtual_to_ego=mount.tolist())),indent=2))
    return dict(step=step.name,source_packet=str(packet),source_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),
                note='Only mount required by native exporter recovered; no inference/control/simulation fabricated.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256((RUN/'run_report.json').read_bytes()).hexdigest()
    report=_hugsim_expand(json.loads((RUN/'run_report.json').read_text()))
    view=OUT/'source_view';view.mkdir(exist_ok=True)
    for name in ['run_report.json','scene_manifest.json','insert_request.json','arm_request.json']:
        if not (view/name).exists():(view/name).symlink_to(RUN/name)
    worker=view/'uniad_worker';worker.mkdir(exist_ok=True)
    recovery_file=OUT/'metadata_recovery.json'
    recovered=_hugsim_expand(json.loads(recovery_file.read_text())) if recovery_file.exists() else []
    for row in report['observations']:
        step=RUN/f"uniad_worker/step_{row['index']:03d}"
        if not (step/'optimization.json').exists():continue
        target=worker/step.name
        if (step/'sensor_metadata.json').exists():
            if not target.exists():target.symlink_to(step)
            continue
        target.mkdir(exist_ok=True)
        for source in step.iterdir():
            if not (target/source.name).exists():(target/source.name).symlink_to(source)
        meta=target/'sensor_metadata.json'
        if not meta.exists():
            recovered.append(recovered_metadata(step,meta))
    recovery_file.write_text(json.dumps(recovered,indent=2))
    sys.path[:0]=[str(P28.parent/'phase11'),str(ROOT/'integrations/sr-hugsim'),str(ROOT/'waymo23-native-import')]
    terminal=terminal_event(report)
    (OUT/'terminal_event.json').write_text(json.dumps(terminal,indent=2))
    plot(report,terminal)
    from comparison_video import encode
    if not (RUN/'heading_fix_scene552_3CAM.mp4').exists():
        video=encode(RUN,10,len(report['observations']),filename='heading_fix_scene552_3CAM.mp4')
        (OUT/'video_report.json').write_text(json.dumps(video,indent=2))
    export=OUT/'native'
    for name,args,artifact in [
        ('native_eval_export.py',['--run',str(view),'--out',str(export)],'data.pkl'),
        ('native_eval_clouds.py',['--run',str(view),'--out',str(export)],'clouds.npz'),
        ('native_eval_score.py',['--export',str(export)],'native_score.json')]:
        if not (export/artifact).exists():subprocess.run([sys.executable,str(P28/name),*args],check=True)
    xy=np.array([np.asarray(o['closed_ego_to_world'])[:2,3] for o in report['observations']])
    summary=dict(simulation_status=report['status'],outcome='fail_actor_proxy_overlap',
        original_error=report['error'],recorded_controls=len(report['trace']),attempted_controls=report['controls_attempted'],
        observations=len(report['observations']),wall_seconds=report['wall_seconds'],
        distance_xy_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        min_insert_gap_m=min(o['inserted_reference_gap_m'] for o in report['observations']),
        terminal_event=terminal,native_score=_hugsim_expand(json.loads((export/'native_score.json').read_text())),
        report_sha256=digest,metadata_recovered_steps=[r['step'] for r in recovered],
        note='Native planned-trajectory scores do not override executed safety stop. Video ends before the interpolated overlap event. Original report untouched.')
    assert hashlib.sha256((RUN/'run_report.json').read_bytes()).hexdigest()==digest
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def terminal_event(report):
    from native_actor_evaluation import NativeActorEvaluation
    from runtime_configuration import load_vehicle_config
    archive=P28.parent/'phase5/native552_actor_geometry.npz'
    with np.load(archive) as data:ev=NativeActorEvaluation.from_archive(data)
    last=report['observations'][-1];pose=np.array(last['closed_ego_to_world']);frame=last['source_frame'];vehicle=load_vehicle_config()
    baseline=ev.check(pose,frame,vehicle)
    assert baseline==last['all_actor_reference'], 'Archive must reproduce recorded last-frame geometry exactly'
    assert last['closed_speed_mps']==0., 'This replay assumes the documented stationary terminal state'
    # At rest, bounded braking leaves pose unchanged. Match the harness's 11
    # fractional-frame checks exactly; do not assume the event was at frame150.
    rows=[]
    for u in np.linspace(0,1,11):
        check=ev.check(pose,frame+float(u),vehicle)
        rows.append(dict(u=float(u),source_frame=frame+float(u),timestamp_s=(frame+float(u))/10.,check=check))
        if check['overlap']:break
    assert rows[-1]['check']['overlap'], 'Expected original safety stop was not reproduced'
    return dict(last_recorded_source_frame=frame,archive=str(archive),archive_matches_last_observation=True,
        first_sampled_overlap=rows[-1],checks=rows,
        scope='Exact replay of existing actor-box proxy check; not physical mesh-contact certification')


def plot(report,terminal):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hugsim-heading-recovery')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from native_actor_evaluation import NativeActorEvaluation
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for label,directory,color in [('old heading + v2',P28/'runs/v3_branch_patch','tab:orange'),
                                  ('heading fix + v2',RUN,'tab:blue')]:
        r=_hugsim_expand(json.loads((directory/'run_report.json').read_text()));obs=r['observations']
        xy=np.array([np.asarray(o['closed_ego_to_world'])[:2,3] for o in obs]);t=np.array([o['timestamp']-8 for o in obs])
        axes[0].plot(xy[:,0],xy[:,1],'.-',label=label,color=color,markersize=2)
        axes[1].plot(t,[o['closed_speed_mps'] for o in obs],label=label,color=color)
        axes[2].plot([r['timestamp']-8 for r in r['trace']],[np.rad2deg(r['state_before'][4]) for r in r['trace']],label=label,color=color)
    axes[0].set(xlabel='world x, m',ylabel='world y, m',title='Executed route (different stop policies)');axes[0].set_aspect('equal')
    axes[1].set(xlabel='simulation time, s',ylabel='speed, m/s',title='Fixed arm stops at 4.6s')
    axes[2].set(xlabel='simulation time, s',ylabel='steering, degrees',title='Countersteering begins; safety braking holds angle')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.suptitle('Heading fix improved steering, but run failed actor-proxy overlap; no pass')
    fig.tight_layout();fig.savefig(OUT/'trajectory_overlay.png',dpi=150);plt.close(fig)
    with np.load(terminal['archive']) as data:ev=NativeActorEvaluation.from_archive(data)
    last=report['observations'][-1];pose=np.asarray(last['closed_ego_to_world']);frame=terminal['first_sampled_overlap']['source_frame']
    from scipy.spatial import ConvexHull
    local=np.array([[a*1.5,b*.8,z,1.] for a in [-1,1] for b in [-1,1] for z in [0.,1.6]])
    vertices=(pose@local.T).T[:,:2]
    ego=vertices[ConvexHull(vertices).vertices];ego=np.vstack([ego,ego[0]])
    fig,ax=plt.subplots(figsize=(8,7));ax.plot(ego[:,0],ego[:,1],'b-',label='stopped ego proxy')
    for group,index,poly,z0,z1 in ev.geometry(frame):
        if group=='RigidNodes' and index==6:
            poly=np.vstack([poly,poly[0]]);ax.plot(poly[:,0],poly[:,1],'r-',label='native vehicle #6 proxy')
    ax.set_aspect('equal');ax.grid(alpha=.3);ax.legend();ax.set(xlabel='world x, m',ylabel='world y, m',title=f'First sampled actor-proxy overlap at source frame {frame:.1f}')
    fig.tight_layout();fig.savefig(OUT/'terminal_overlap_bev.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
