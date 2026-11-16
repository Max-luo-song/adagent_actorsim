"""Configured Waymo23 experiment: optionally run, then collect and compare."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw


def first_difference(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('planning trajectory shapes differ')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite trajectory')
    d = np.linalg.norm(b-a, axis=1)
    return dict(per_point_m=d.tolist(),mean_m=float(d.mean()),max_m=float(d.max()))


def run_trial(base, steps, remove, log_path):
    command = [sys.executable,str(base/'waymo23-native-import/uniad_closed_fixed.py'),
               '--steps',str(steps)]
    if remove is not None:
        command += ['--remove',str(remove)]
    result = None
    with log_path.open('w') as log:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
            if line.startswith('OUTPUT:'):
                result = Path(line.split(':',1)[1].strip())
        code = process.wait()
    if code or result is None:
        raise RuntimeError('trial failed; see '+str(log_path))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--run',action='store_true',help='rerun both trials before collecting')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config['scene']!='waymo23' or config['model']!='uniad':
        raise ValueError('this entry currently supports waymo23 + uniad only')
    steps = int(config['steps']);removed = int(config['remove_instance'])
    if not 1<=steps<=10 or removed<0:
        raise ValueError('invalid steps or instance ID')
    base = Path(os.environ['BASE']).resolve()
    out = base/'experiments'/('waymo23_delete_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    print('EXPERIMENT:',out,flush=True)
    if args.run:
        original = run_trial(base,steps,None,out/'baseline.log')
        edited = run_trial(base,steps,removed,out/'edited.log')
    else:
        original = base/'outputs'/config['baseline_result']
        edited = base/'outputs'/config['edited_result']
    reports, traces = [], []
    for folder,expected_edit in [(original,None),(edited,removed)]:
        report = json.loads((folder/'closed_report.json').read_text())
        trace = json.loads((folder/'trace.json').read_text())
        if (report['controller']!='forward_right_heading_fixed_v1'
                or report['steps']!=steps or len(trace)!=steps
                or report['removed_instance']!=expected_edit
                or report['command']!=2 or report['dt']!=.1):
            raise ValueError('trial settings mismatch: '+str(folder))
        if report['scene_checkpoint_loads']!=1 or report['uniad_model_loads']!=1:
            raise ValueError('unexpected load counts')
        reports.append(report);traces.append(trace)
    pairs = [];first_infos = [];first_images = []
    for step in range(steps):
        frames = []
        for folder in (original,edited):
            with (folder/f'step_{step:03d}'/'observation.pkl').open('rb') as f:
                obs, info = pickle.load(f)
            if step==0:
                first_infos.append(info);first_images.append(obs['rgb']['CAM_FRONT'])
            frames.append(Image.fromarray(obs['rgb']['CAM_FRONT']))
        left,right = frames
        if left.size!=right.size:
            raise ValueError('image sizes differ')
        canvas = Image.new('RGB',(left.width*2,left.height+24),'white')
        canvas.paste(left,(0,24));canvas.paste(right,(left.width,24))
        draw = ImageDraw.Draw(canvas)
        draw.text((5,5),f'Original | t={step*.1:.1f}s',fill='black')
        draw.text((left.width+5,5),f'Remove {removed} | t={step*.1:.1f}s',fill='black')
        pairs.append(canvas)
    ia,ib = first_infos
    for key in ('ego_pos','ego_rot','ego_velo','ego_steer','timestamp','command'):
        if not np.allclose(ia[key],ib[key],atol=1e-8,rtol=0):
            raise ValueError('initial states differ: '+key)
    if set(ia['cam_params'])!=set(ib['cam_params']):
        raise ValueError('camera sets differ')
    for name in ia['cam_params']:
        if not np.allclose(ia['cam_params'][name]['l2c'],ib['cam_params'][name]['l2c']):
            raise ValueError('camera extrinsics differ')
        if ia['cam_params'][name]['intrinsic']!=ib['cam_params'][name]['intrinsic']:
            raise ValueError('camera intrinsics differ')
    a,b = traces
    metrics = first_difference(a[0]['trajectory'],b[0]['trajectory'])
    diff = np.abs(first_images[1].astype(np.int16)-first_images[0].astype(np.int16))
    result = dict(stage='edited-scene closed-loop integration comparison',
                  scene='waymo23',model='uniad',removed_instance=removed,steps=steps,
                  baseline=str(original),edited=str(edited),initial_states_and_calibration_match=True,
                  first_front_changed_pixels=int((diff.max(2)>5).sum()),
                  first_trajectory_difference=metrics,
                  first_acceleration_delta=float(b[0]['applied_control']['acc']-a[0]['applied_control']['acc']),
                  final_ego_distance_m=float(np.linalg.norm(np.asarray(b[-1]['ego_after'])-a[-1]['ego_after'])),
                  rear_view_quality_validated=False,repeatability_validated=False,
                  driving_quality_validated=False,collision_and_scoring_validated=False)
    pairs[0].save(out/'first_frame_comparison.png')
    pairs[0].save(out/'comparison.gif',save_all=True,append_images=pairs[1:],duration=100,loop=0)
    resolved = dict(config,baseline_result=original.name,edited_result=edited.name)
    (out/'experiment.json').write_text(json.dumps(resolved,indent=2))
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    hashes = {}
    for name in ('uniad_closed_fixed.py','trajectory_control.py','uniad_single.py','live_env.py'):
        hashes[name] = hashlib.sha256((base/'waymo23-native-import'/name).read_bytes()).hexdigest()
    (out/'current_code_sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(result,indent=2));print('EXPERIMENT:',out)


if __name__=='__main__':
    main()
