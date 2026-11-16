import contextlib
from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np


def compare_trajectories(a,b):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2:
        raise ValueError('trajectory shapes differ')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite trajectory')
    return float(np.linalg.norm(a-b,axis=1).mean())


def replay_model(model,namespace):
    from five_uniad import ORDER,adapt_camera_modules
    from uniad_single import TAIL
    source = Path(os.environ['REPLAY_SOURCE'])
    job = Path(os.environ['REPLAY_JOB'])
    changes = adapt_camera_modules(model)
    old = "_cameras = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT',\n            'CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']"
    marker = "_data.pop('raw_imgs',None)"
    if TAIL.count(old)!=1 or TAIL.count(marker)!=1:
        raise ValueError('unexpected preprocessing template')
    tail = TAIL.replace(old,'_cameras = '+repr(ORDER))
    tail = tail.replace('virtual_rear_views=True','virtual_rear_views=False')
    tail = tail.replace(marker,marker+
        "\n_data['img_metas'][0][0]['scene_token'] = _os.environ['REPLAY_TOKEN']")
    code = compile(tail,'recorded_observation_replay','exec')
    packets = sorted(source.glob('step_*/observation.pkl'))
    recorded = json.loads((source/'trace.json').read_text())
    if not packets or len(packets)!=len(recorded):
        raise ValueError('recorded packet/trace counts differ')
    modes = {}
    for mode in ('history','new_scene_each_frame'):
        rows = []
        for i,packet in enumerate(packets):
            folder = job/mode/f'step_{i:03d}'
            folder.mkdir(parents=True)
            token = 'replay_history_v1' if mode=='history' else f'replay_fresh_{i}'
            os.environ['REPLAY_TOKEN'] = token
            os.environ['UNIAD_PACKET'] = str(packet)
            os.environ['UNIAD_SMOKE_OUTPUT'] = str(folder)
            scope = dict(namespace,model=model)
            with (folder/'inference.log').open('w') as log:
                with contextlib.redirect_stdout(log):
                    exec(code,scope)
            report = scope['_report']
            traj = report['trajectory']
            row = dict(step=i,scene_token=token,trajectory=traj,
                       last_forward_m=float(traj[-1][1]),
                       recorded_mean_difference_m=compare_trajectories(
                           traj,recorded[i]['trajectory']))
            rows.append(row)
            print(mode,'step',i,'last_forward_m',
                  round(row['last_forward_m'],4),flush=True)
        modes[mode] = rows
    comparison = []
    for a,b in zip(modes['history'],modes['new_scene_each_frame']):
        comparison.append(dict(
            step=a['step'],
            history_last_forward_m=a['last_forward_m'],
            new_scene_last_forward_m=b['last_forward_m'],
            mean_plan_difference_m=compare_trajectories(
                a['trajectory'],b['trajectory']),
            history_vs_recorded_mean_m=a['recorded_mean_difference_m']))
    report = dict(
        stage='recorded-observation temporal diagnostic',
        source=str(source),scene_rendered=False,uniad_model_loads=1,
        camera_adaptations=changes,
        reset_method='unique scene_token triggers built-in scene-change paths',
        reset_scope_not_independently_validated=True,
        identical_saved_inputs_used=True,closed_loop_rerun=False,
        comparison=comparison,trajectories=modes)
    (job/'replay_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(comparison,indent=2))
    print('REPLAY OUTPUT:',job)


def main():
    import five_uniad
    base = Path(os.environ['BASE'])
    source = Path(os.environ['REPLAY_SOURCE']).resolve()
    report = json.loads((source/'closed_report.json').read_text())
    if report.get('camera_count')!=5:
        raise ValueError('expected a five-camera recorded run')
    packet = source/'step_000/observation.pkl'
    if not packet.is_file():
        raise FileNotFoundError(packet)
    job = base/'outputs'/(
        'five_temporal_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    job.mkdir(parents=True)
    (job/'observation.pkl').symlink_to(packet)
    os.environ['REPLAY_JOB'] = str(job)
    os.environ['FIVE_JOB'] = str(job)
    five_uniad.infer_five = replay_model
    print('REPLAY OUTPUT:',job,flush=True)
    five_uniad.infer()


if __name__=='__main__':
    main()
