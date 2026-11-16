"""Post-hoc frozen-model comparisons; placement proxies never enter inference."""

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
pass  # Path is provided by the portable boundary above
import pickle
import subprocess
import sys
import numpy as np
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'waymo23-native-import'))
from evaluate_preview_vehicle import reference_polygon,polygon_iou,box_polygon
from box_geometry import BOX_CONVENTION,box_corners


def overlay(image, info, camera, data, reference):
    from PIL import ImageDraw
    from asset_geometry import source_corners
    from evaluate_preview_vehicle import B,T
    draw=ImageDraw.Draw(image);params=info['cam_params'][camera];intr=params['intrinsic']
    k=np.array([[intr['W']/(2*np.tan(intr['fovx']/2)),0,intr['cx']],
                [0,intr['H']/(2*np.tan(intr['fovy']/2)),intr['cy']],[0,0,1.]])
    def wire(points,edges,color):
        cam=np.column_stack([points,np.ones(len(points))])@np.asarray(params['l2c']).T
        projected=cam[:,:3]@k.T
        for a,b in edges:
            if min(cam[a,2],cam[b,2])<=.1:continue
            xy=projected[[a,b],:2]/projected[[a,b],2:3]
            if np.isfinite(xy).all() and np.abs(xy).max()<1e5:draw.line([tuple(xy[0]),tuple(xy[1])],fill=color,width=2)
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for box,label in zip(data.get('boxes',[]),data.get('labels',[])):
        if label==0:wire(box_corners(box),edges,'cyan')
    pose=np.eye(4);pose[:3,:3]=info['explicit_l2g_r_mat'];pose[:3,3]=info['explicit_l2g_t']
    points=np.column_stack([source_corners(reference),np.ones(8)])@(np.linalg.inv(pose)@B@T).T
    wire(points[:,:3],[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)],'yellow')
    return image


def match(data,truth,prefix=''):
    boxes=np.asarray(data.get(prefix+'boxes',np.empty((0,7))))
    scores=np.asarray(data.get(prefix+'scores',[])).reshape(-1)
    labels=np.asarray(data.get(prefix+'labels',[])).reshape(-1)
    if len(boxes)!=len(scores) or len(boxes)!=len(labels):raise ValueError('Misaligned model detections')
    candidates=[(polygon_iou(box_polygon(b),truth),i) for i,b in enumerate(boxes) if labels[i]==0]
    if not candidates:return dict(matched=False,iou=0.,car_count=0)
    iou,index=max(candidates);poly=box_polygon(boxes[index])
    # Both vectors follow the longitudinal dimension in their corner contracts.
    a=truth[1]-truth[0];b=poly[2]-poly[1]
    angle=np.degrees(np.arccos(np.clip(abs(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))
    return dict(matched=bool(iou>=.25),iou=float(iou),index=int(index),score=float(scores[index]),
        center_error_m=float(np.linalg.norm(poly.mean(0)-truth.mean(0))),heading_error_mod180_deg=float(angle),
        car_count=len(candidates),matches_at_iou025=sum(v>=.25 for v,_ in candidates))


def region_occupancy(array,truth):
    masks=np.asarray(array).reshape(1,5,200,200)[0]!=0
    xy=(np.indices((200,200))[::-1].transpose(1,2,0)-100)*.5+.25
    region=np.all((xy>=truth.min(0)-.5)&(xy<=truth.max(0)+.5),axis=-1)
    return [int((m&region).sum()) for m in masks]


def analyze(output,report):
    from PIL import Image,ImageDraw
    output=Path(output);all_results=[]
    by_scene={}
    for arm in report['inference']:by_scene.setdefault(str(arm['scene']),{})[arm['role']]=arm
    for scene,arms in by_scene.items():
        if set(arms)!={'baseline','inserted','moved'}:raise ValueError('Incomplete matched arms')
        baseline=arms['baseline']['records'];results={}
        for role,arm in arms.items():
            ref=_hugsim_expand(json.loads((Path(arm['sequence']).parent/'asset_reference.json').read_text()));frames=[]
            for i,record in enumerate(arm['records']):
                with open(record['packet'],'rb') as f:obs,info=pickle.load(f)
                with open(baseline[i]['packet'],'rb') as f:base_obs,base_info=pickle.load(f)
                for key in ('timestamp','source_frame','ego_pos','ego_rot','ego_velo','ego_steer','command','accelerate','steer_rate','explicit_l2g_r_mat','explicit_l2g_t'):
                    if not np.allclose(info[key],base_info[key]):raise ValueError('Counterfactual metadata mismatch: '+key)
                for camera in info['cam_params']:
                    if obs['rgb'][camera].shape!=base_obs['rgb'][camera].shape:raise ValueError('Image shape changed')
                    for key in ('l2c','v2c'):
                        if not np.allclose(info['cam_params'][camera][key],base_info['cam_params'][camera][key]):raise ValueError('Camera calibration changed')
                    if info['cam_params'][camera]['intrinsic']!=base_info['cam_params'][camera]['intrinsic']:raise ValueError('Camera intrinsic changed')
                truth=reference_polygon(ref,info)
                with np.load(record['perception']) as p,np.load(baseline[i]['perception']) as q:
                    data={k:p[k] for k in p.files};base={k:q[k] for k in q.files}
                if str(data['box_convention'])!=BOX_CONVENTION:raise ValueError('Wrong raw box convention')
                delta=data['trajectory']-base['trajectory']
                frames.append(dict(frame=record['frame'],timestamp=record['timestamp'],retained=match(data,truth),
                    candidate=match(data,truth,'candidate_'),baseline_at_same_reference=match(base,truth),
                    regional_occupancy=region_occupancy(data['occupancy'],truth),baseline_regional_occupancy=region_occupancy(base['occupancy'],truth),
                    occupancy_changed_cells=int(np.count_nonzero(data['occupancy']!=base['occupancy'])),
                    image_changed_pixels={name:int(np.any(obs['rgb'][name]!=base_obs['rgb'][name],axis=-1).sum()) for name in obs['rgb']},
                    plan_mean_l2_delta_m=float(np.linalg.norm(delta,axis=-1).mean()),plan_max_l2_delta_m=float(np.linalg.norm(delta,axis=-1).max()),
                    plan_endpoint=data['trajectory'][-1].tolist(),baseline_plan_endpoint=base['trajectory'][-1].tolist(),
                    endpoint_delta_xy_m=delta[-1].tolist(),reference_lidar_xy=truth.tolist()))
            results[role]=frames
        # Render all three arms in the camera that actually shows the insertion.
        camera='CAM_FRONT_RIGHT' if scene=='172' else 'CAM_FRONT'
        for i in range(len(baseline)):
            tiles=[]
            for role in ('baseline','inserted','moved'):
                rec=arms[role]['records'][i]
                image=Image.open(Path(rec['packet']).parent/(camera+'.png')).convert('RGB')
                with open(rec['packet'],'rb') as f:_,info=pickle.load(f)
                with np.load(rec['perception']) as p:data={k:p[k] for k in p.files}
                reference=_hugsim_expand(json.loads((Path(arms[role]['sequence']).parent/'asset_reference.json').read_text()))
                image=overlay(image,info,camera,data,reference)
                draw=ImageDraw.Draw(image);item=results[role][i]
                draw.rectangle((0,0,image.width,48),fill='black')
                draw.text((8,6),f'{scene} {role} {camera} frame {rec["frame"]}',fill='white')
                draw.text((8,25),f'cyan=retained cars; yellow=asset proxy; IoU={item["retained"]["iou"]:.3f}',fill='white')
                tiles.append(image)
            canvas=Image.new('RGB',(sum(t.width for t in tiles),max(t.height for t in tiles)))
            x=0
            for tile in tiles:canvas.paste(tile,(x,0));x+=tile.width
            canvas.save(output/f'{scene}_comparison_{i:03d}.png')
        movement=[]
        for i,(first,second) in enumerate(zip(results['inserted'],results['moved'])):
            entry=dict(frame=first['frame'],reference_center_shift_xy_m=(np.mean(second['reference_lidar_xy'],0)-np.mean(first['reference_lidar_xy'],0)).tolist())
            if 'index' in first['retained'] and 'index' in second['retained']:
                with np.load(arms['inserted']['records'][i]['perception']) as p,np.load(arms['moved']['records'][i]['perception']) as q:
                    change=q['boxes'][second['retained']['index'],:2]-p['boxes'][first['retained']['index'],:2]
                entry.update(best_box_center_shift_xy_m=change.tolist(),shift_error_m=float(np.linalg.norm(change-entry['reference_center_shift_xy_m'])),
                    warning='Best spatial matches may be different identities; this is not track certification')
            movement.append(entry)
        all_results.append(dict(scene=scene,camera=camera,arms=results,movement_response=movement))
        ffmpeg=BASE/'code/HUGSIM/.pixi/envs/default/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
        if ffmpeg.is_file():
            subprocess.run([str(ffmpeg),'-y','-v','error','-framerate','2','-i',str(output/f'{scene}_comparison_%03d.png'),
                '-frames:v',str(len(baseline)),'-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(output/f'{scene}_comparison.mp4')],check=True,timeout=90)
    summary=dict(schema='unified_uniad_posthoc_v1',training=False,checkpoint_unchanged=report['checkpoint_unchanged'],
        matching_rule='car label 0, oriented BEV IoU >= 0.25, common to both scenes',
        reference_is_gaussian_support_proxy=True,identity_certified=False,closed_loop_run=False,
        formal_evaluation_admitted=False,unseen_scene_generalization_proven=False,scenes=all_results)
    (output/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    lines=['# 统一接口接入冻结 UniAD：诊断结果','',
        '不训练；N1 暂停；未执行控制。本实验为相同记录位姿下的三组回放，不是闭环绕行验证。',
        '参考框来自资产几何支撑代理，仅在推理完成后用于匹配；没有送入模型。IoU≥0.25仅说明空间匹配，不证明目标身份。','',
        '| 场景 | 组别 | 保留框匹配/帧 | 候选框匹配/帧 | 无车基线同位置匹配/帧 | 平均规划差异(m) |',
        '|---|---|---:|---:|---:|---:|']
    for scene in all_results:
        for role,frames in scene['arms'].items():
            lines.append(f'| {scene["scene"]} | {role} | {sum(f["retained"]["matched"] for f in frames)}/{len(frames)} | {sum(f["candidate"]["matched"] for f in frames)}/{len(frames)} | {sum(f["baseline_at_same_reference"]["matched"] for f in frames)}/{len(frames)} | {np.mean([f["plan_mean_l2_delta_m"] for f in frames]):.4f} |')
    lines+=['','基线也匹配时，不能把重叠框直接认定为新插入车辆。位置、朝向误差、附近占据与逐帧规划终点见 analysis.json。',
        '规划变化只能证明模型响应；准确识别、合理制动与安全绕行仍需独立验证。两个已知场景不能证明未见场景泛化。',
        'comparison.mp4 为5帧记录轨迹回放，以2fps慢放（原时间间隔0.1秒），不是闭环视频。青框为原始保留car框，黄框为资产参考几何；无车组也显示黄色参考位置。']
    (output/'analysis.md').write_text('\n'.join(lines)+'\n')
    return summary
