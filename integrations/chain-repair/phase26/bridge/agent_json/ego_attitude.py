"""Metric pose correction using Waymo gravity, not a 2D image rotation.

Waymo raw world is ENU. The renderer world is first-ego, so world-up must be
rotated by the FIRST pose, not assumed to be [0,0,1] in renderer coordinates.
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

pass  # Path is provided by the portable boundary above
import json
import math
import subprocess
import numpy as np
from PIL import Image,ImageDraw
from waymo_adapter import rigid

POLICIES=('initial_ego','recorded_up','gravity_level')


def gravity_up(spec):
    source=spec.source
    if source.get('scene_adapter',{}).get('coordinate_system','waymo_z_up_meters')!='waymo_z_up_meters':
        raise ValueError('gravity alignment requires Waymo ENU pose provenance')
    origin=rigid(np.loadtxt(Path(source['inputs'])/'ego_pose'/f"{source['start_timestep']:03d}.txt"))
    return origin[:3,:3].T@np.array([0.,0.,1.])


def corrected_pose(value,policy,up_source,recorded=None):
    pose=rigid(value).copy()
    if policy not in POLICIES:raise ValueError('unknown attitude_policy')
    if policy=='initial_ego':return pose
    up=np.asarray(up_source,dtype=float) if policy=='gravity_level' else rigid(recorded)[:3,2]
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-8:raise ValueError('invalid up vector')
    up=up/np.linalg.norm(up)
    forward=pose[:3,0]-up*np.dot(up,pose[:3,0])
    if np.linalg.norm(forward)<1e-6:raise ValueError('heading is parallel to requested up')
    forward/=np.linalg.norm(forward);left=np.cross(up,forward)
    pose[:3,:3]=np.column_stack([forward,left,up])
    return rigid(pose)


def horizon_metrics(packet,up_source):
    up=np.asarray(up_source,dtype=float);up/=np.linalg.norm(up)
    pose=np.asarray(packet['ego_to_world']);tilt=float(np.degrees(np.arccos(np.clip(pose[:3,2]@up,-1,1))))
    cameras={}
    for name,cam in packet['cameras'].items():
        c2w=np.asarray(cam['camera_to_world']);k=np.asarray(cam['K']);g=c2w[:3,:3].T@up
        line=np.linalg.inv(k).T@g
        angle=math.degrees(math.atan2(-line[0],line[1]));angle=(angle+90)%180-90
        cameras[name]=dict(projected_horizon_tilt_degrees=float(angle),horizon_line=line.tolist(),
            optical_axis_elevation_degrees=float(np.degrees(np.arcsin(np.clip(g[2],-1,1)))))
    return dict(body_up_tilt_degrees=tilt,gravity_up_source=up.tolist(),cameras=cameras,
        metric='projection of gravity-orthogonal plane; not an image-line detector')


def pose_for_sample(compiled,pose,index):
    sample=compiled['samples'][index]
    return corrected_pose(pose,compiled.get('attitude_policy','initial_ego'),
        compiled['gravity_up_source'],sample['recorded_ego_to_world'])


def present_attitude_comparison(runs,out,backend):
    """Same time/position, original geometry renders, old attitude versus level."""
    from ego_trajectory import ffmpeg_path
    old=next((r for r in runs if r['trajectory'].get('attitude_policy')=='initial_ego'),None)
    fixed=next((r for r in runs if r['trajectory'].get('attitude_policy')=='gravity_level'),None)
    if old is None or fixed is None:return None
    a=_hugsim_expand(json.loads(Path(old['manifest']).read_text()));b=_hugsim_expand(json.loads(Path(fixed['manifest']).read_text()))
    if len(a['frames'])!=len(b['frames']):return None
    pairs=list(zip(a['frames'],b['frames']))
    if not all(x['new']['timestamp_s']==y['new']['timestamp_s'] and
        np.allclose(np.asarray(x['new']['ego_to_world'])[:3,3],np.asarray(y['new']['ego_to_world'])[:3,3],atol=1e-9,rtol=0)
        for x,y in pairs):return None
    folder=Path(out)/'attitude_comparison';folder.mkdir();frames=folder/'video_frames';frames.mkdir()
    order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT');metrics=[];checks=[]
    for i,(x,y) in enumerate(pairs):
        before=x['new'];after=y['new'];up=fixed['trajectory']['gravity_up_source']
        m0=horizon_metrics(before,up);m1=horizon_metrics(after,up)
        checks.append(m1['body_up_tilt_degrees']<1e-4)
        sheet=Image.new('RGB',(1440,800),(20,20,20));draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f'3D attitude correction | {backend} | same ego position/time | left / front / right',fill='white')
        for row,(label,packet,metric) in enumerate([('BEFORE: initial tilted ego plane',before,m0),('AFTER: body aligned to Waymo gravity',after,m1)]):
            y0=40+row*360
            tilt=metric['cameras']['CAM_FRONT']['projected_horizon_tilt_degrees']
            draw.text((8,y0),f"{label} | time={packet['timestamp_s']:.1f}s | front predicted horizon tilt={tilt:.3f}deg",fill='white')
            for col,name in enumerate(order):
                c=packet['cameras'][name];other=after['cameras'][name] if row==0 else before['cameras'][name]
                checks.append(np.allclose(c['K'],other['K'],atol=1e-9,rtol=0) and np.allclose(c['camera_to_ego'],other['camera_to_ego'],atol=1e-9,rtol=0))
                with Image.open(c['rgb']) as source:
                    im=source.convert('RGB');im.thumbnail((480,320));sheet.paste(im,(col*480,y0+28))
        draw.text((8,770),'Camera extrinsics preserved. Raw RGB/depth re-rendered from corrected 3D poses; no image rotation.',fill='white')
        sheet.save(frames/f'{i:05d}.png');metrics.append(dict(timestamp_s=before['timestamp_s'],before=m0,after=m1))
    ffmpeg=ffmpeg_path();video=folder/'horizon_before_after.mp4'
    subprocess.run([str(ffmpeg),'-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-n','-framerate',str(fixed['trajectory']['fps']),'-i',str(frames/'%05d.png'),
        '-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
    subprocess.run([str(ffmpeg),'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    result=dict(passed=all(checks),video=str(video),metrics=metrics,position_and_time_preserved=True,
        camera_mount_preserved=True,image_rotation=False,execution_backend=backend,visual_quality_certified=False)
    (folder/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def checked_horizon_metrics(packet,up_source,tolerance=.5):
    result=horizon_metrics(packet,up_source)
    result['max_camera_horizon_tilt_degrees']=max(abs(c['projected_horizon_tilt_degrees']) for c in result['cameras'].values())
    result['camera_roll_tolerance_degrees']=float(tolerance)
    result['gravity_alignment_passed']=result['body_up_tilt_degrees']<1e-4 and result['max_camera_horizon_tilt_degrees']<=tolerance
    return result


def audit_task_attitude(output,policy):
    """Check actual saved packets for every view, not just new ego poses."""
    output=Path(output);packets=_hugsim_expand(json.loads((output/'preview_index.json').read_text()));rows=[]
    for packet in packets:
        metric=packet.get('attitude')
        rows.append(dict(request_id=packet['request_id'],source_frame=packet['source_frame'],timestamp_s=packet['timestamp_s'],
            body_up_tilt_degrees=metric['body_up_tilt_degrees'] if metric else None,
            cameras=metric['cameras'] if metric else {},
            aligned=bool(metric and metric['body_up_tilt_degrees']<1e-4 and all(abs(c['projected_horizon_tilt_degrees'])<=.5 for c in metric['cameras'].values()))))
    applicable=policy=='gravity_level' and bool(rows)
    report=dict(policy=policy,applicable=applicable,passed=all(row['aligned'] for row in rows) if applicable else True,
        camera_roll_tolerance_degrees=.5,observations=len(rows),metrics=rows,
        max_camera_horizon_tilt_degrees=max((abs(c['projected_horizon_tilt_degrees']) for row in rows for c in row['cameras'].values()),default=None))
    (output/'camera_attitude.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def compare_camera_levels(observations,output,backend):
    """Ordinary raw vs level views: same ego anchor/time and camera mounts."""
    raw=[p for p in observations if p['action'] in ('preview','observe') and p.get('attitude_policy')=='initial_ego']
    level=[p for p in observations if p['action'] in ('preview','observe') and p.get('attitude_policy')=='gravity_level']
    for before in raw:
        for after in level:
            if not (before['scene']==after['scene'] and before['timestamp_s']==after['timestamp_s'] and before['edit_revision']==after['edit_revision'] and before['history_token']==after['history_token'] and np.array_equal(np.asarray(before['ego_to_world'])[:3,3],np.asarray(after['ego_to_world'])[:3,3])):continue
            same_mount=all(np.allclose(before['cameras'][name]['K'],camera['K'],atol=1e-9,rtol=0) and np.allclose(before['cameras'][name]['camera_to_ego'],camera['camera_to_ego'],atol=1e-9,rtol=0) for name,camera in after['cameras'].items())
            pixels={}
            for name,camera in after['cameras'].items():
                with Image.open(before['cameras'][name]['rgb']) as a,Image.open(camera['rgb']) as b:pixels[name]=int(np.any(np.asarray(a)!=np.asarray(b),axis=-1).sum())
            metric=after['attitude'];aligned=metric['body_up_tilt_degrees']<1e-4 and all(abs(c['projected_horizon_tilt_degrees'])<=.5 for c in metric['cameras'].values())
            result=dict(passed=bool(same_mount and aligned and (any(pixels.values()) if backend=='framework_renderer' else True)),
                before=before['attitude'],after=metric,ego_position_and_time_preserved=True,camera_mount_and_K_preserved=bool(same_mount),
                rgb_changed_pixels=pixels,gpu_render_executed=backend=='framework_renderer',image_rotation=False,
                note='Rig rotation preserves ego anchor and mount geometry; optical centers move with the rigid camera lever arms.')
            (Path(output)/'camera_level_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            return result
    return None
