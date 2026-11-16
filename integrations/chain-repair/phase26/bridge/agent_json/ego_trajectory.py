"""Finite, timed ego paths in the existing source-world coordinate contract.

Rendering uses caller poses, never moves the scene, trains, or invokes control.
Each waypoint is an actual render sample; interpolation is deliberately absent.
"""

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
import subprocess
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image, ImageDraw
from waymo_adapter import rigid, frame_index


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value)-set(required)-set(optional):
        raise ValueError('missing or unknown trajectory fields: '+','.join(sorted(required)))


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value):
        raise ValueError('trajectory values must be finite numbers')
    return float(value)


def recorded_pose(spec, frame):
    source=spec.source; folder=Path(source['inputs'])/'ego_pose'; start=source['start_timestep']
    return rigid(np.linalg.inv(rigid(np.loadtxt(folder/f'{start:03d}.txt'))) @
                 rigid(np.loadtxt(folder/f'{start+frame:03d}.txt')))


def compile_trajectory(value, spec):
    keys(value, {'schema','coordinate_frame','start_frame','waypoints'}, {'attitude_policy'})
    if value['schema'] != 'ego_trajectory_v1':raise ValueError('expected ego_trajectory_v1')
    mode=value['coordinate_frame']
    if mode not in ('start_ego','source_world'):raise ValueError('coordinate_frame must be start_ego or source_world')
    start=value['start_frame']; points=value['waypoints']
    if type(start) is not int or not 0<=start<spec.frames:raise ValueError('start_frame outside scene')
    if not isinstance(points,list) or not 2<=len(points)<=300:raise ValueError('trajectory requires 2..300 samples')
    from ego_attitude import POLICIES,gravity_up,corrected_pose
    policy=value.get('attitude_policy','initial_ego')
    if policy not in POLICIES:raise ValueError('unknown attitude_policy')
    up=gravity_up(spec)
    anchor=recorded_pose(spec,start); compiled=[]; previous=-1.
    for point in points:
        if mode=='start_ego':
            keys(point,{'time_s','forward_m','left_m','yaw_degrees'})
        else:keys(point,{'time_s','ego_to_world'})
        t=number(point['time_s'])
        if t<=previous or (not compiled and t!=0):raise ValueError('times must start at zero and strictly increase')
        # Exact source cadence avoids ambiguous flooring and video time distortion.
        offset=round(t*spec.fps)
        if abs(t*spec.fps-offset)>1e-6:raise ValueError('sample times must align to scene frame cadence')
        if compiled and offset!=compiled[-1]['frame_offset']+1:raise ValueError('provide each scene frame; sparse interpolation is unsupported')
        frame=start+offset; stamp=start/spec.fps+t
        if frame>=spec.frames:raise ValueError('trajectory extends beyond scene time')
        frame_index(stamp,spec.fps,spec.frames)
        if mode=='start_ego':
            x=number(point['forward_m']); y=number(point['left_m']); yaw=number(point['yaw_degrees'])
            angle=np.deg2rad(yaw); c,s=np.cos(angle),np.sin(angle)
            delta=np.array([[c,-s,0,x],[s,c,0,y],[0,0,1,0],[0,0,0,1]],dtype=float)
            pose=rigid(anchor@delta)
        else:pose=rigid(point['ego_to_world'])
        recorded=recorded_pose(spec,frame);nominal=pose.copy()
        pose=corrected_pose(nominal,policy,up,recorded)
        compiled.append(dict(timestamp_s=stamp,time_s=t,source_frame=frame,frame_offset=offset,
            ego_to_world=pose.tolist(),motion_ego_to_world=nominal.tolist(),recorded_ego_to_world=recorded.tolist()))
        previous=t
    poses=np.asarray([x['ego_to_world'] for x in compiled]); dt=np.diff([x['time_s'] for x in compiled])
    speeds=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)/dt
    return dict(schema='compiled_ego_trajectory_v1',coordinate_frame=mode,
        attitude_policy=policy,gravity_up_source=up.tolist(),gravity_authority='Waymo ENU raw world z rotated into first-ego scene frame',
        axes='x_forward_y_left_z_up',world='first_waymo_ego',sample_count=len(compiled),
        fps=float(spec.fps),duration_s=previous,start_frame=start,samples=compiled,
        maximum_sampled_speed_mps=float(speeds.max()),interpolation='none; one render per supplied sample',
        mutation_performed=False,control_executed=False,closed_loop_driving=False,
        collision_checked=False,road_compliance_checked=False,visual_quality_certified=False)


def ffmpeg_path():
    import imageio_ffmpeg
    path=Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not path.is_file():raise FileNotFoundError('ffmpeg binary unavailable')
    return path


def trajectory_metrics(packets):
    """Measure saved packet poses, independently of caller waypoint labels."""
    anchor=np.asarray((packets[0]['baseline'] or packets[0]['new'])['ego_to_world'])
    inverse=np.linalg.inv(anchor); results=[]
    for pair in packets:
        new=np.asarray(pair['new']['ego_to_world']); local=inverse@new
        row=dict(timestamp_s=pair['new']['timestamp_s'],new_forward_m=float(local[0,3]),
            new_left_m=float(local[1,3]),new_yaw_degrees=float(np.degrees(np.arctan2(local[1,0],local[0,0]))))
        if pair['baseline']:
            base=np.asarray(pair['baseline']['ego_to_world']); recorded=inverse@base; delta=np.linalg.inv(base)@new
            row.update(recorded_forward_m=float(recorded[0,3]),recorded_left_m=float(recorded[1,3]),
                separation_m=float(np.linalg.norm(new[:3,3]-base[:3,3])),
                relative_yaw_degrees=float(np.degrees(np.arctan2(delta[1,0],delta[0,0]))))
        results.append(row)
    return results


def draw_trajectory_panel(draw, metrics, index, y, *, execution=False):
    # Fixed shared scale and full paths make motion and comparison unambiguous.
    points=[(r['new_left_m'],r['new_forward_m']) for r in metrics]
    baseline=[(r['recorded_left_m'],r['recorded_forward_m']) for r in metrics if 'recorded_left_m' in r]
    combined=points+baseline
    low=np.min(combined,axis=0); high=np.max(combined,axis=0); center=(low+high)/2
    scale=min(560/max(high[0]-low[0],1),140/max(high[1]-low[1],1))
    def pixel(point):return (int(320-(point[0]-center[0])*scale),int(y+110-(point[1]-center[1])*scale))
    draw.text((10,y+8),'Top view | forward: up | left: left',fill='white')
    for path,color in [(baseline,(180,180,180)),(points,(0,230,230))]:
        if len(path)>1:draw.line([pixel(point) for point in path],fill=color,width=3)
        if path:
            x,z=pixel(path[min(index,len(path)-1)]);draw.ellipse((x-5,z-5,x+5,z+5),fill=color)
    row=metrics[index]
    draw.text((660,y+20),'GRAY: JSON target | CYAN: executed Motion path' if execution else 'GRAY: recorded path | CYAN: JSON new path',fill='white')
    draw.text((660,y+50),f"{'Executed' if execution else 'New'} pose from start: forward={row['new_forward_m']:.2f}m left={row['new_left_m']:.2f}m yaw={row['new_yaw_degrees']:.1f}deg",fill=(0,230,230))
    if 'separation_m' in row:
        draw.text((660,y+80),f"Against {'target' if execution else 'recorded'} pose now: distance={row['separation_m']:.2f}m yaw difference={row['relative_yaw_degrees']:.1f}deg",fill='white')
    draw.text((660,y+115),'Metrics use actual rendered packet poses. Units: metres / degrees.',fill='white')
    draw.text((660,y+145),'iLQR state feedback executed | no UniAD inference / collision evaluation' if execution else 'Trajectory preview; controller and collision checks are not executed.',fill='white')


def make_video(packets, output, fps, backend):
    folder=Path(output); frames=folder/'video_frames'; frames.mkdir()
    order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT'); w,h=480,320
    metrics=trajectory_metrics(packets)
    (folder/'pose_comparison.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    for index,pair in enumerate(packets):
        rows=[('recorded ego trajectory',pair['baseline'])] if pair['baseline'] else []
        rows.append(('JSON new ego trajectory',pair['new']))
        panel_y=48+(h+32)*len(rows)
        sheet=Image.new('RGB',(w*3,panel_y+200),(20,20,20)); draw=ImageDraw.Draw(sheet)
        draw.text((8,6),f'Ego trajectory preview | {backend} | no controller / no closed loop',fill='white')
        draw.text((8,24),f"frame={pair['new']['source_frame']} time={pair['new']['timestamp_s']:.3f}s | left / front / right",fill='white')
        for row,(label,packet) in enumerate(rows):
            y=48+row*(h+32);draw.text((8,y+8),label,fill='white')
            for col,cam in enumerate(order):
                with Image.open(packet['cameras'][cam]['rgb']) as source:
                    im=source.convert('RGB');im.thumbnail((w,h));sheet.paste(im,(col*w,y+32))
        draw_trajectory_panel(draw,metrics,index,panel_y)
        sheet.save(frames/f'{index:05d}.png')
    video=folder/'ego_trajectory_comparison.mp4'; ffmpeg=ffmpeg_path()
    subprocess.run([str(ffmpeg),'-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-n','-framerate',str(fps),'-i',str(frames/'%05d.png'),
        '-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
    subprocess.run([str(ffmpeg),'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    return str(video)


def preview_trajectory(bridge, args, request_id, compiled):
    # All trajectory validation and ffmpeg availability precede GPU renders.
    ffmpeg_path()
    version={k:args[k] for k in ('expected_generation','expected_revision')}
    objects=_hugsim_expand(json.loads(json.dumps(bridge.session.objects)))
    out=bridge.output/'trajectories'/request_id;out.mkdir(parents=True,exist_ok=False)
    (out/'trajectory.json').write_text(json.dumps(compiled,indent=2,allow_nan=False))
    from ego_attitude import corrected_pose
    def expected_pose(sample,kind):
        value=sample['recorded_ego_to_world'] if kind=='baseline' else sample['ego_to_world']
        if kind=='baseline' and args.get('level_recorded_baseline',False):
            return corrected_pose(value,'gravity_level',compiled['gravity_up_source'],sample['recorded_ego_to_world']).tolist()
        return value
    packets=[]; manifest=dict(completed=False,frames=packets,execution_backend=bridge.execution_backend)
    try:
        for index,sample in enumerate(compiled['samples']):
            pair=dict(baseline=None,new=None)
            for kind,field in ([('baseline','recorded_ego_to_world')] if args.get('compare_recorded',True) else [])+[('new','ego_to_world')]:
                policy=('gravity_level' if args.get('level_recorded_baseline',False) else 'initial_ego') if kind=='baseline' else compiled['attitude_policy']
                obs=dict(version,timestamp_s=sample['timestamp_s'],ego_to_world=expected_pose(sample,kind),attitude_policy=policy)
                pair[kind]=bridge.observe(obs,f'{request_id}_{index:04d}_{kind}',preview=True)
                from ego_attitude import checked_horizon_metrics
                pair[kind]['attitude']=checked_horizon_metrics(pair[kind],compiled['gravity_up_source'])
            packets.append(pair)
            print(f'Ego trajectory preview {index+1}/{compiled["sample_count"]}',flush=True)
        audit=dict(generation_preserved=bridge.session.generation==version['expected_generation'],
            revision_preserved=bridge.session.revision==version['expected_revision'],
            managed_objects_preserved=bridge.session.objects==objects,
            pre_run_preserved=not bridge.session.editor.started,
            history_cleared=getattr(bridge.session.backend,'_last',None) is None,
            sample_count_matches=len(packets)==compiled['sample_count'],
            packet_poses_and_times_match=all(
                packet['source_frame']==sample['source_frame'] and packet['timestamp_s']==sample['timestamp_s'] and
                packet['history_token']==version['expected_generation'] and packet['edit_revision']==version['expected_revision'] and
                packet['preview_only'] and not packet['observation_history_committed'] and
                np.allclose(packet['ego_to_world'],expected_pose(sample,kind),rtol=0,atol=1e-9)
                for sample,pair in zip(compiled['samples'],packets)
                for kind,field in [('baseline','recorded_ego_to_world'),('new','ego_to_world')]
                if (packet:=pair[kind]) is not None))
        audit['attitude_policy_matches']=all(
            np.allclose(np.asarray(pair['new']['ego_to_world'])[:3,2],compiled['gravity_up_source'],atol=1e-8,rtol=0)
            if compiled['attitude_policy']=='gravity_level' else True for pair in packets)
        manifest['audit']=audit
        manifest.update(level_recorded_baseline=args.get('level_recorded_baseline',False),attitude_policy=compiled['attitude_policy'],gravity_up_source=compiled['gravity_up_source'])
        if not all(audit.values()):raise RuntimeError('ego trajectory preview audit failed')
        video=make_video(packets,out,compiled['fps'],bridge.execution_backend)
        manifest.update(completed=True,video=video)
    finally:
        (out/'packets.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return dict(trajectory=compiled,audit=audit,manifest=str(out/'packets.json'),video=video,
        rendered_packets=len(packets)*(2 if args.get('compare_recorded',True) else 1),
        preview_only=True,observation_history_committed=False,
        generation=bridge.session.generation,edit_revision=bridge.session.revision)
