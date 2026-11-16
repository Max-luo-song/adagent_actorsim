"""New scene variant: move the swapped native car forward, retaining the ego view."""

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
import copy
import json
pass  # Path is provided by the portable boundary above

import numpy as np

import workflow as w
from contracts import read, write, sha

OLD = w.P26 / 'novel_view_position_swap_v1/scene551_ego_native3'
OFFSET = 15.0


def verify(out):
    plan = read(out/'plan.json')
    if plan['code'] != w.pins():
        raise ValueError('Far-car implementation changed; retain this run and use a new version')
    for a in list(plan['assets'].values()) + list(plan['source_input_assets'].values()):
        w.verify_asset(a)
    return plan


def prepare():
    out = w.SOURCE
    if out.exists():
        verify(out)
        return dict(prepared=True, reused=True, output=str(out))
    parent = read(OLD/'plan.json')
    prior = read(OLD/'preview/report.json')
    if not prior['completed'] or not prior['original_scene_restored']:
        raise ValueError('Completed parent preview required')
    for path, expected in parent['code'].items():
        if sha(path) != expected:
            raise ValueError('Parent code changed: '+path)
    for a in list(parent['assets'].values()) + list(parent['source_input_assets'].values()):
        w.verify_asset(a)
    with np.load(OLD/'swap_program.npz') as packed:
        program = {k: packed[k].copy() for k in packed.files}
    frames = program['frames']
    ego = program['ego_poses']
    prior_car = program['car_trans'][frames].copy()
    # Keep the original ground-path tangent, including its local road grade.
    forward = program['original_ego'][:, :3, 0]
    forward = forward / np.linalg.norm(forward, axis=1)[:, None]
    program['car_trans'][frames] += OFFSET * forward
    car = program['car_trans'][frames]
    local = np.einsum('nji,nj->ni', ego[:, :3, :3], car-ego[:, :3, 3])
    if local[:, 0].min() < 8:
        raise ValueError('Far car would leave the forward viewing region')
    # Check swept sampled footprints against all current native actors, not only ego.
    from scipy.spatial.transform import Rotation
    from shapely.geometry import Polygon
    def footprint(center, rotation, size):
        x, y = size[:2]/2
        corners = np.array([[-x,-y,0],[x,-y,0],[x,y,0],[-x,y,0]])
        return Polygon((corners@rotation.T+center)[:, :2])
    with np.load(parent['assets']['effective_actor_tracks']['path']) as packed:
        tracks = {k: packed[k] for k in packed.files}
    size = tracks['RigidNodes__instances_size'][3]
    gaps = []; other_gaps = []; conflicts = []
    for i, frame in enumerate(frames):
        rotation = Rotation.from_quat(program['original_car_quats'][frame][[1,2,3,0]]).as_matrix()
        poly = footprint(car[i], rotation, size)
        gaps.append(poly.distance(footprint(ego[i,:3,3], ego[i,:3,:3], np.array([3.,1.6]))))
        for group in ['RigidNodes','DeformableNodes','SMPLNodes']:
            for actor in np.flatnonzero(tracks[group+'__instances_fv'][frame]):
                if group == 'RigidNodes' and actor == 3:
                    continue
                q = tracks[group+'__instances_quats'][frame,actor].reshape(-1,4)[0]
                rot = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                gap = poly.distance(footprint(tracks[group+'__instances_trans'][frame,actor], rot, tracks[group+'__instances_size'][actor]))
                other_gaps.append(gap)
                if gap < .1:
                    conflicts.append([int(frame),group,int(actor),float(gap)])
    if min(gaps) < .1 or conflicts:
        raise ValueError('Far-car footprint conflict: '+str(conflicts[:5]))
    dt = np.diff(frames)/10.
    speed = np.linalg.norm(np.diff(car,axis=0)/dt[:,None],axis=1)
    acceleration = np.linalg.norm(np.diff(np.diff(car,axis=0)/dt[:,None],axis=0)/dt[1:,None],axis=1)
    report = dict(schema='far_car_preview_geometry_v1', object_id='native:RigidNodes:3',
        parent_program=w.asset(OLD/'swap_program.npz'), forward_offset_m=OFFSET,
        semantics='Offline initial scene variant; shifted car follows original swapped path plus local forward offset. Ego unchanged. No driving.',
        minimum_ego_target_footprint_gap_m=float(min(gaps)), minimum_other_actor_gap_m=float(min(other_gaps)),
        forward_distance_start_m=float(local[0,0]),forward_distance_end_m=float(local[-1,0]),
        forward_distance_min_m=float(local[:,0].min()),forward_distance_max_m=float(local[:,0].max()),
        speed_min_mps=float(speed.min()),speed_max_mps=float(speed.max()),max_sampled_acceleration_mps2=float(acceleration.max()),
        all_sampled_frames_in_front=True,other_actor_conflicts=conflicts,
        road_compliance_certified=False,background_collision_certified=False,closed_loop_driving=False,
        note='6 second preview range only; no continuity or driving guarantee beyond source frames 0..60')
    out.mkdir(parents=True)
    np.savez_compressed(out/'swap_program.npz',**program)
    write(out/'ego_trajectory.json',read(OLD/'ego_trajectory.json'))
    write(out/'geometry_report.json',report)
    plan = copy.deepcopy(parent)
    plan.update(schema='far_car_preview_v1',output=str(out),code=w.pins(),forward_offset_m=OFFSET)
    plan['assets'].update(swap_program=w.asset(out/'swap_program.npz'),trajectory=w.asset(out/'ego_trajectory.json'),
                          parent_plan=w.asset(OLD/'plan.json'),parent_preview=w.asset(OLD/'preview/report.json'),
                          geometry_report=w.asset(out/'geometry_report.json'))
    write(out/'plan.json',plan)
    return dict(prepared=True,output=str(out),geometry=report)


def preview():
    out = w.SOURCE
    plan = verify(out)
    stage = out/'preview'
    if (stage/'report.json').exists():
        report = read(stage/'report.json')
        assert report['completed']
        w.verify_asset(report['video'])
        for row in report['records']:
            w.verify_asset(row['image'])
        return dict(reused=True,video=report['video'])
    if stage.exists():
        raise ValueError('Incomplete preview retained: '+str(stage))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no rendering performed')
    from PIL import Image,ImageDraw
    import imageio_ffmpeg
    import subprocess
    from preview import TranslationSwap,sha_tensor
    from smpl_runtime import canonical_layer,assets
    module = w.provider()
    if module.family_pins() != plan['provider_code']:
        raise ValueError('Conditioned renderer changed')
    module.CONTEXT = plan['conditioning']
    task = plan['source_task']
    module.framework.bootstrap(task['renderer_bundle'])
    with canonical_layer(assets(),task['renderer_bundle']):
        renderer = module.ConditionedRenderer(task['scene_manifest'],task['renderer_bundle'])
    with np.load(out/'swap_program.npz') as packed:
        program = {k:packed[k].copy() for k in packed.files}
    prior = read(OLD/'preview/report.json')
    old = {(r['source_frame'],r['camera_id'],r['variant']):r for r in prior['records']}
    records = []; cleanup = dict(closed=False,actor_restored=False)
    stage.mkdir();(stage/'frames').mkdir()
    try:
        swap = TranslationSwap(renderer.backend,program)
        actor_hashes = {n:sha_tensor(m.instances_trans) for n,m in renderer.trainer.models.items() if hasattr(m,'instances_trans')}
        restored = []
        for i,frame in enumerate(plan['frames']):
            canvas = Image.new('RGB',(1440,1060),'#151b22');draw = ImageDraw.Draw(canvas)
            draw.text((10,8),f'Scene551 | frame={frame} t={frame/10:.1f}s | SAME ego view, native car moved forward 15m',fill='white')
            draw.text((10,32),'LEFT: previous close car                    RIGHT: farther car / source model, before NEW distillation',fill='white')
            draw.text((10,55),'Offline edited scene preview; no control execution. Previous teacher/student results are not reused for this new edit.',fill='white')
            for row,cam in enumerate([1,0,2]):
                old_image = old[frame,cam,'swapped']['image'];w.verify_asset(old_image)
                data = renderer.frame(program['ego_poses'][i],frame,cam)
                with swap.at(frame,True),torch.no_grad():
                    rgb = (renderer.rgb(data)['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
                path = stage/f'{frame:03d}_cam{cam}_swapped.png';Image.fromarray(rgb).save(path)
                for col,p in enumerate([old_image['path'],path]):
                    with Image.open(p) as image:
                        image = image.convert('RGB');image.thumbnail((720,290));canvas.paste(image,(col*720,130+row*300))
                    draw.text((col*720+8,108+row*300),f'camera {cam}',fill='white')
                original = old[frame,cam,'original'];w.verify_asset(original['image']);records.append(original)
                records.append(dict(source_frame=frame,camera_id=cam,variant='swapped',edit_revision=1,
                                    ego_to_world=program['ego_poses'][i].tolist(),image=w.asset(path)))
            canvas.save(stage/'frames'/f'{i:04d}.png')
            write(stage/'progress.json',dict(completed_frames=i+1,total_frames=len(plan['frames'])))
            print(f'Far-car preview {i+1}/{len(plan["frames"])}',flush=True)
        cleanup['actor_restored'] = all(sha_tensor(renderer.trainer.models[n].instances_trans)==h for n,h in actor_hashes.items())
        for cam in [0,1,2]:
            data = renderer.frame(program['original_ego'][0],plan['frames'][0],cam)
            with torch.no_grad():rgb = (renderer.rgb(data)['rgb'].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
            with Image.open(old[plan['frames'][0],cam,'original']['image']['path']) as image:
                restored.append(np.array_equal(rgb,np.asarray(image)))
        cleanup['initial_pixels_restored'] = all(restored)
        if not cleanup['actor_restored'] or not all(restored):
            raise ValueError('Original actor or image restoration failed')
        video = stage/'close_vs_far_native_car_3CAM.mp4';ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff,'-v','error','-n','-framerate','10','-i',str(stage/'frames/%04d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(video)],check=True)
        subprocess.run([ff,'-v','error','-threads','1','-i',str(video),'-f','null','-'],check=True)
        result = dict(completed=True,both_ego_and_vehicle_positions_changed=True,original_scene_restored=True,
                      video=w.asset(video),records=records,full_decode_passed=True,frames=len(plan['frames']),
                      gpu_executed=True,training_performed=False,closed_loop_driving=False,
                      geometry_report=w.asset(out/'geometry_report.json'))
        write(stage/'report.json',result)
        return {k:v for k,v in result.items() if k!='records'}
    finally:
        renderer.close();cleanup['closed']=True;write(stage/'cleanup.json',cleanup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','preview']);args=parser.parse_args()
    print(json.dumps(prepare() if args.stage=='prepare' else preview(),indent=2))
