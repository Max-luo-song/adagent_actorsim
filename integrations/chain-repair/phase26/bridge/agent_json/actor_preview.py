"""Paired actual renders for a native pedestrian root trajectory."""

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

import json,re,subprocess
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw
from ego_trajectory import recorded_pose,ffmpeg_path
from ego_attitude import corrected_pose,gravity_up


def make_video(frames,output,fps,backend,object_id):
    folder=Path(output)/'video_frames';folder.mkdir()
    order=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT');w,h=480,320
    for i,pair in enumerate(frames):
        sheet=Image.new('RGB',(1440,816),(20,20,20));draw=ImageDraw.Draw(sheet)
        draw.text((8,8),f'{object_id} | fixed gravity-level camera | {backend}',fill='white')
        draw.text((8,26),f"time={pair['new']['timestamp_s']:.1f}s | source-derived body motion | no driving",fill='white')
        for row,(kind,label) in enumerate([('baseline','Original programmed pedestrian paths (other edits retained)'),('new','JSON pedestrian path')]):
            y=48+row*352;draw.text((8,y+6),label,fill='white')
            for col,cam in enumerate(order):
                with Image.open(pair[kind]['cameras'][cam]['rgb']) as source:
                    im=source.convert('RGB');im.thumbnail((w,h));sheet.paste(im,(col*w,y+32))
        draw.text((8,760),f"Root separation: {pair['root_separation_m']:.2f}m | RGB changed pixels: {sum(pair['changed_pixels'].values())}",fill='white')
        draw.text((8,785),'Raw camera PNGs, depth and per-frame root evidence are saved alongside this video.',fill='white')
        sheet.save(folder/f'{i:05d}.png')
    video=Path(output)/'pedestrian_trajectory_comparison.mp4';ffmpeg=str(ffmpeg_path())
    subprocess.run([ffmpeg,'-v','error','-threads','1','-filter_threads','1','-filter_complex_threads','1','-n','-framerate',str(fps),'-i',str(folder/'%05d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','20',str(video)],check=True)
    subprocess.run([ffmpeg,'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    return str(video)


def make_closeup(frames,output,fps,object_id):
    """A fixed crop of saved front-camera RGB, enclosing both root paths."""
    camera='CAM_FRONT';points=[]
    with Image.open(frames[0]['new']['cameras'][camera]['rgb']) as image:width,height=image.size
    for pair in frames:
        for kind in ('baseline','new'):
            packet=pair[kind];cal=packet['cameras'][camera]
            center=next(s['root_world_m'] for s in packet['actor_trajectories']['states'] if s['object_id']==object_id)
            world=np.array(center)+np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-2,2)])
            inverse=np.linalg.inv(cal['camera_to_world']);view=world@inverse[:3,:3].T+inverse[:3,3]
            good=view[:,2]>.1
            if good.any():
                pixels=view[good]@np.array(cal['K']).T;points.extend(pixels[:,:2]/pixels[:,2,None])
    crop=(0,0,width,height)
    if points:
        pixels=np.array(points);low=np.floor(pixels.min(0)-20).astype(int);high=np.ceil(pixels.max(0)+20).astype(int)
        box=(max(0,int(low[0])),max(0,int(low[1])),min(width,int(high[0])),min(height,int(high[1])))
        if box[2]>box[0] and box[3]>box[1]:crop=box
    out=Path(output);folder=out/'closeup_frames';folder.mkdir()
    for i,pair in enumerate(frames):
        sheet=Image.new('RGB',(1280,520),(20,20,20));draw=ImageDraw.Draw(sheet)
        for col,(kind,label) in enumerate([('baseline','Original path'),('new','JSON new path')]):
            draw.text((col*640+8,8),f"{label} | t={pair[kind]['timestamp_s']:.1f}s | {object_id}",fill='white')
            with Image.open(pair[kind]['cameras'][camera]['rgb']) as image:
                im=image.convert('RGB').crop(crop);im.thumbnail((640,480));sheet.paste(im,(col*640,32))
        sheet.save(folder/f'{i:05d}.png')
        if i==len(frames)-1:sheet.save(out/'pedestrian_closeup_comparison.png')
    video=out/'pedestrian_closeup_comparison.mp4';ffmpeg=str(ffmpeg_path())
    subprocess.run([ffmpeg,'-v','error','-threads','1','-filter_threads','1','-n','-framerate',str(fps),'-i',str(folder/'%05d.png'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','18',str(video)],check=True)
    subprocess.run([ffmpeg,'-v','error','-threads','2','-i',str(video),'-f','null','-'],check=True)
    return str(video),list(crop)


def preview_actor(bridge,args,request_id):
    from agent_bridge import BridgeError
    spec=bridge.session.spec;manager=bridge.session.editor.actor_paths
    oid=args['object_id'];match=re.fullmatch(r'native:SMPLNodes:(0|[1-9][0-9]*)',oid) if isinstance(oid,str) else None
    if match is None or int(match.group(1)) not in manager.programs:raise BridgeError('invalid_actor_trajectory','target requires an active native SMPL trajectory')
    start,end,camera=[args[k] for k in ('start_frame','end_frame','camera_source_frame')]
    if any(type(v) is not int or not 0<=v<spec.frames for v in (start,end,camera)) or not 1<=end-start<=120:raise BridgeError('invalid_actor_trajectory','preview requires 2..121 in-range source frames')
    render_fps=args.get('render_fps',spec.fps)
    if isinstance(render_fps,bool) or not isinstance(render_fps,(int,float)) or render_fps not in (spec.fps,20,30) or render_fps<spec.fps:raise BridgeError('invalid_actor_trajectory','render_fps requires source FPS, 20 or 30')
    if render_fps!=spec.fps and any(e['motion'] is None for e in manager.programs.values()):raise BridgeError('invalid_actor_trajectory','subframe preview requires smooth_v1 for all programmed actors')
    count=int(round((end-start)/spec.fps*render_fps))+1
    if count>361:raise BridgeError('invalid_actor_trajectory','preview exceeds 361 render samples')
    timestamps=np.linspace(start/spec.fps,end/spec.fps,count)
    index=int(match.group(1));valid=manager.model.instances_fv[start:end+1,index]
    if not bool(valid.all()):raise BridgeError('invalid_actor_trajectory','preview includes unavailable source body animation')
    from actor_trajectories import sample_path
    if any(sample_path(manager.programs[index]['plan'],t) is None for t in timestamps):raise BridgeError('invalid_actor_trajectory','preview must stay inside an active path interval')
    previous=None
    if 'comparison_baseline' in args:
        from actor_motion_review import load_baseline
        try:previous=load_baseline(args['comparison_baseline'],spec,oid,start,end,camera)
        except (ValueError,OSError,KeyError,TypeError) as exc:raise BridgeError('invalid_motion_baseline',str(exc)) from exc
    motion=manager.programs[index]['motion']
    motion_quality=None if motion is None else {key:motion[key] for key in ('min_playback_rate','max_playback_rate','last_animation_frame','target_distance_m','velocity_jump_before_mps','velocity_jump_after_mps','joint_roughness_before','joint_roughness_after')}
    ffmpeg_path();policy=args.get('attitude_policy','gravity_level')
    try:pose=corrected_pose(recorded_pose(spec,camera),policy,gravity_up(spec),recorded_pose(spec,camera)).tolist()
    except ValueError as exc:raise BridgeError('invalid_attitude_policy',str(exc)) from exc
    version={k:args[k] for k in ('expected_generation','expected_revision')};fingerprint=manager.animation_fingerprint()
    roots=manager.trans_tensor.detach().clone();quats=manager.quat_tensor.detach().clone();programs=set(manager.programs)
    out=bridge.output/'actor_trajectories'/request_id;out.mkdir(parents=True,exist_ok=False)
    frames=[];manifest=dict(completed=False,frames=frames,object_id=oid,camera_source_frame=camera,execution_backend=bridge.execution_backend,attitude_policy=policy,render_fps=float(render_fps),source_fps=float(spec.fps),motion_profile=manager.programs[index]['plan']['motion_profile'],motion_quality=motion_quality)
    def save():(out/'packets.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    try:
        for sample_index,timestamp in enumerate(timestamps):
            obs=dict(version,timestamp_s=float(timestamp),ego_to_world=pose,attitude_policy=policy)
            with manager.original_paths():baseline=bridge.observe(obs,f'{request_id}_{sample_index:04d}_baseline',preview=True)
            new=bridge.observe(obs,f'{request_id}_{sample_index:04d}_new',preview=True)
            original=next(s for s in baseline['actor_trajectories']['states'] if s['object_id']==oid)
            target=next(s for s in new['actor_trajectories']['states'] if s['object_id']==oid)
            evidence=[e for e in new['actor_trajectories']['render_evidence'] if e['object_id']==oid]
            changed={}
            for cam in new['cameras']:
                with Image.open(baseline['cameras'][cam]['rgb']) as a,Image.open(new['cameras'][cam]['rgb']) as b:
                    changed[cam]=int(np.any(np.asarray(a)!=np.asarray(b),axis=-1).sum())
            frames.append(dict(baseline=baseline,new=new,root_separation_m=float(np.linalg.norm(np.array(original['root_world_m'])-target['root_world_m'])),changed_pixels=changed,render_evidence=evidence))
            save();print(f'Pedestrian trajectory preview {sample_index+1}/{count}',flush=True)
        audit=dict(source_body_and_appearance_unchanged=fingerprint==manager.animation_fingerprint(),
            root_tensors_restored_after_counterfactual=bool(manager.torch.equal(roots,manager.trans_tensor) and manager.torch.equal(quats,manager.quat_tensor)),
            programs_preserved=programs==set(manager.programs),generation_preserved=bridge.session.generation==version['expected_generation'],
            revision_preserved=bridge.session.revision==version['expected_revision'],pre_run_preserved=not bridge.session.editor.started,
            history_cleared=getattr(bridge.session.backend,'_last',None) is None,
            paired_camera_time_matches=all(p['baseline']['timestamp_s']==p['new']['timestamp_s'] and p['new']['source_frame']==int(np.floor(timestamps[i]*spec.fps+1e-8)) and abs(p['new']['timestamp_s']-timestamps[i])<1e-9 and
                np.array_equal(p['baseline']['ego_to_world'],p['new']['ego_to_world']) and all(
                    np.array_equal(p['baseline']['cameras'][c]['K'],p['new']['cameras'][c]['K']) and
                    np.array_equal(p['baseline']['cameras'][c]['camera_to_world'],p['new']['cameras'][c]['camera_to_world']) for c in p['new']['cameras']) for i,p in enumerate(frames)),
            actual_gaussian_root_transform_verified=all(p['render_evidence'] and all(e['gaussian_count']>0 and e['source_frame_valid'] and e['proxy_error_m']<1e-4 and e['render_root_error_m']<1e-4 and e['rigid_shape_error_m']<1e-4 for e in p['render_evidence']) for p in frames),
            smooth_motion_constraints_verified=motion is None or (motion['velocity_jump_after_mps']<1e-3 and motion['joint_roughness_after']<=motion['joint_roughness_before']+1e-7),
            rendered_rgb_changed=any(any(p['changed_pixels'].values()) for p in frames))
        manifest.update(audit=audit,maximum_root_separation_m=max(p['root_separation_m'] for p in frames))
        if not all(audit.values()):raise RuntimeError('pedestrian trajectory preview audit failed; inspect packets.json')
        video=make_video(frames,out,render_fps,bridge.execution_backend,oid)
        closeup,crop=make_closeup(frames,out,render_fps,oid)
        comparison=None
        if previous is not None:
            from actor_motion_review import compare_motion
            comparison=compare_motion(previous,frames,out,render_fps,crop)
        manifest.update(completed=True,video=video,closeup_video=closeup,motion_comparison=comparison,front_fixed_crop_pixels=crop,baseline_scope='all programmed pedestrian root trajectories; all other edits retained')
    finally:save()
    return dict(audit=audit,motion_quality=motion_quality,motion_comparison=comparison,manifest=str(out/'packets.json'),video=video,closeup_video=closeup,rendered_packets=2*len(frames),
        maximum_root_separation_m=manifest['maximum_root_separation_m'],preview_only=True,observation_history_committed=False,
        generation=bridge.session.generation,edit_revision=bridge.session.revision,gait_retimed=manager.programs[index]['motion'] is not None,render_fps=float(render_fps),motion_profile=manager.programs[index]['plan']['motion_profile'])
