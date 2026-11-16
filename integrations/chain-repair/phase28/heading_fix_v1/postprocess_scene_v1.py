"""Produce native scores, rendered video and trajectory plot after this run."""

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
import os
pass  # Path is provided by the portable boundary above
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
P28=HERE.parent
ROOT=P28.parents[5]


def main():
    run=Path(_hugsim_expand(json.loads((HERE/'scene552_request_v1.json').read_text()))['output'])
    report=_hugsim_expand(json.loads((run/'run_report.json').read_text()))
    out=run/'analysis'
    out.mkdir(exist_ok=True)
    export=out/'native'
    for script,args in [('native_eval_export.py',['--run',str(run),'--out',str(export)]),
                        ('native_eval_clouds.py',['--run',str(run),'--out',str(export)]),
                        ('native_eval_score.py',['--export',str(export)])]:
        subprocess.run([sys.executable,str(P28/script),*args],check=True)
    sys.path.insert(0,str(ROOT/'integrations/sr-hugsim'))
    from comparison_video import encode
    video=encode(run,10,len(report['observations']),filename='heading_fix_scene552_3CAM.mp4')
    (out/'video_report.json').write_text(json.dumps(video,indent=2))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/hugsim-heading-scene-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for label,directory,color in [('old v2-on',P28/'runs/v3_branch_patch','tab:orange'),
                                  ('heading fix + v2',run,'tab:blue')]:
        rep=_hugsim_expand(json.loads((directory/'run_report.json').read_text()))
        obs=rep['observations'];xy=np.array([np.asarray(o['closed_ego_to_world'])[:2,3] for o in obs])
        t=np.array([o['timestamp']-obs[0]['timestamp'] for o in obs])
        axes[0].plot(xy[:,0],xy[:,1],'.-',label=label,color=color,markersize=2)
        axes[1].plot(t,[o['closed_speed_mps'] for o in obs],label=label,color=color)
        axes[2].plot([r['timestamp']-obs[0]['timestamp'] for r in rep['trace']],
                     [np.rad2deg(r['state_before'][4]) for r in rep['trace']],label=label,color=color)
    asset=_hugsim_expand(json.loads((run/'asset_reference.json').read_text()));w,l,h=asset['asset_wlh'];bottom=asset['local_bottom_y']
    local=np.array([[x,bottom,z,1.] for x,z in [(-l/2,-w/2),(l/2,-w/2),(l/2,w/2),(-l/2,w/2),(-l/2,-w/2)]])
    placement=next(x for x in asset['placements'] if x['case']=='heading_A')
    box=(np.array(placement['asset_to_source'])@local.T).T
    axes[0].plot(box[:,0],box[:,1],'k--',label='insert proxy');axes[0].set_aspect('equal')
    axes[0].set(xlabel='world x, m',ylabel='world y, m',title='Executed trajectories')
    axes[1].set(xlabel='simulation time, s',ylabel='speed, m/s',title='Executed speed')
    axes[2].set(xlabel='simulation time, s',ylabel='steering, degrees',title='Steering: negative = left')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.suptitle('Single-arm mechanism observation; termination differs from v3; not patch on/off certification')
    fig.tight_layout();fig.savefig(out/'trajectory_overlay.png',dpi=150);plt.close(fig)
    obs=report['observations'];xy=np.array([np.asarray(o['closed_ego_to_world'])[:2,3] for o in obs])
    summary=dict(status=report['status'],outcome=report.get('interaction_outcome'),termination_reason=report.get('termination_reason'),
        wall_seconds=report['wall_seconds'],controls=len(report['trace']),distance_xy_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
        min_inserted_gap_m=min(o['inserted_reference_gap_m'] for o in obs),
        actor_proxy_overlaps=sum(bool(o['all_actor_reference']['overlap']) for o in obs),
        native_vehicle_proxy_overlaps=sum(bool(o['native_vehicle_reference']['overlap']) for o in obs),
        score=_hugsim_expand(json.loads((export/'native_score.json').read_text())),
        note='New single-arm observation; no new-code patch-off comparison. Physical road boundaries not certified.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
