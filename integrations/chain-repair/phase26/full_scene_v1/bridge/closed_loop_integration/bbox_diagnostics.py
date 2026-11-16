"""CPU postprocessing only: asset proxy / raw detection / planner proposal audit.

Inputs are saved sensor packets and controller receipts, never live control.
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

import argparse,csv,json,pickle,os,subprocess
pass  # Path is provided by the portable boundary above
import numpy as np
from PIL import Image,ImageDraw

GREEN=(70,240,100);RED=(255,85,85);ORANGE=(255,175,35)
EDGES=[(i,(i//4)*4+(i+1)%4) for i in range(8)]+[(i,i+4) for i in range(4)]
CAMERAS=('CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT')
CONVENTION='mmdet3d_legacy_lidar_xyz_dims_clockwise_yaw_bottom_center_v1'


def box_corners(box):
    b=np.asarray(box,dtype=float)
    if b.shape not in ((7,),(9,)) or not np.isfinite(b).all() or np.any(b[3:6]<=0):
        raise ValueError('Invalid bottom-centered detection box')
    c,s=np.cos(b[6]),np.sin(b[6])
    xy=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*b[3:5]/2
    xy=xy@np.array([[c,-s],[s,c]])+b[:2]
    return np.vstack([np.c_[xy,np.full(4,b[2])],np.c_[xy,np.full(4,b[2]+b[5])]])


def target_corners(ref,info):
    if len(ref['placements'])!=1:raise ValueError('Expected one pinned static inserted target')
    contract=ref['geometry_contract']
    if contract['schema']!='asset_geometry_x_longitudinal_y_down_z_lateral_wlh_v1':
        raise ValueError('Unsupported asset geometry convention')
    w,l,h=ref['asset_wlh'];bottom=ref['local_bottom_y']
    local=np.array([[x,y,z,1] for y in (bottom,bottom-h)
        for x,z in ((-l/2,-w/2),(l/2,-w/2),(l/2,w/2),(-l/2,w/2))])
    native=(local@np.asarray(ref['placements'][0]['asset_to_native']).T)[:,:3]
    # Native x-right,y-down,z-forward -> model-world x-right,y-forward,z-up.
    world=native@np.array([[1,0,0],[0,0,-1],[0,1,0]])
    return (world-np.asarray(info['explicit_l2g_t']))@np.asarray(info['explicit_l2g_r_mat'])


def projected_segments(corners,params,near=.1):
    points=np.c_[corners,np.ones(len(corners))]@np.asarray(params['l2c']).T
    intr=params['intrinsic'];w,h=intr['W'],intr['H']
    fx=w/(2*np.tan(intr['fovx']/2));fy=h/(2*np.tan(intr['fovy']/2))
    segments=[]
    for i,j in EDGES:
        a,b=points[i,:3].copy(),points[j,:3].copy()
        if max(a[2],b[2])<near:continue
        if a[2]<near:a=a+(b-a)*(near-a[2])/(b[2]-a[2])
        if b[2]<near:b=b+(a-b)*(near-b[2])/(a[2]-b[2])
        uv=np.array([[fx*p[0]/p[2]+intr['cx'],fy*p[1]/p[2]+intr['cy']] for p in (a,b)])
        # Clip against the image rectangle, including edges spanning the image.
        delta=uv[1]-uv[0];lo,hi=0.,1.
        for axis,limit in ((0,w-1),(1,h-1)):
            if abs(delta[axis])<1e-12:
                if not 0<=uv[0,axis]<=limit:lo,hi=1.,0.;break
            else:
                bounds=sorted(((0-uv[0,axis])/delta[axis],(limit-uv[0,axis])/delta[axis]))
                lo=max(lo,bounds[0]);hi=min(hi,bounds[1])
        if lo<=hi:segments.append((uv[0]+lo*delta,uv[0]+hi*delta))
    return segments


def compare(target,boxes,scores,labels,threshold=.4,candidate_floor=.05,gate=3.):
    boxes=np.asarray(boxes);scores=np.asarray(scores);labels=np.asarray(labels)
    valid=np.isfinite(boxes).all(1)&np.isfinite(scores)&(boxes[:,3:6]>=.2).all(1)&(boxes[:,3:6]<=30).all(1)&np.isin(labels,[0,1,2,3,4])
    center=target[:4,:2].mean(0)
    def nearest(mask):
        indices=np.flatnonzero(valid&mask)
        if not len(indices):return None
        j=int(indices[np.argmin(np.linalg.norm(boxes[indices,:2]-center,axis=1))])
        distance=float(np.linalg.norm(boxes[j,:2]-center))
        dimensions=np.array([np.linalg.norm(target[2]-target[1]),np.linalg.norm(target[1]-target[0]),np.linalg.norm(target[4]-target[0])])
        forward=target[1,:2]-target[0,:2];forward/=np.linalg.norm(forward)
        predicted_forward=np.array([np.sin(boxes[j,6]),np.cos(boxes[j,6])])
        return dict(candidate_index=j,center_error_xy_m=distance,score=float(scores[j]),
            within_proximity_gate=distance<=gate,velocity_xy_mps=boxes[j,7:9].tolist(),
            static_target_speed_error_mps=float(np.linalg.norm(boxes[j,7:9])),
            dimension_error_wlh_m=(boxes[j,3:6]-dimensions).tolist(),
            unsigned_axis_heading_error_rad=float(np.arccos(np.clip(abs(forward@predicted_forward),0,1))),
            identity_confirmed=False)
    return dict(nearest_candidate=nearest(scores>=candidate_floor),
        nearest_retained_detection=nearest(scores>=threshold),
        detection_threshold=threshold,candidate_floor=candidate_floor,proximity_gate_m=gate,
        association='per-frame nearest vehicle center; not identity tracking or recall',identity_confirmed=False)


def planner_shapes(optimization,memory):
    guard=optimization.get('guard',{});receipt=guard.get('proposal_geometry_receipt')
    if receipt is not None:return receipt['rows'],'actual_guard_arrays'
    # Historical runs predate receipts. Keep their provenance explicit.
    obstacles={o['candidate_index']:o for o in memory.get('obstacles',[])};rows=[]
    for row in guard.get('evidence_tiers',[]):
        o=obstacles.get(row['candidate_index'])
        if o is None:continue
        rows.append(dict(candidate_index=row['candidate_index'],box=o['box'],
            geometry=row['geometry'],tier='hard' if row['hard_constraint'] else 'soft',
            disk=row['geometry']=='heading_free_disk_motion_envelope',
            radius_m=row.get('geometry_radius_m'),margin_m=row['effective_margin_m']))
    return rows,'historical_memory_and_tier_audit_proxy'


def render_frame(packet,info,target,boxes,scores,labels,shapes,source,index,comparison,optimization,replay_only=False):
    canvas=Image.new('RGB',(1440,760),(18,20,25));draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f'Step {index:03d} | GREEN asset reference proxy | RED raw vehicle detections >= 0.05 | ORANGE planner base',fill='white')
    draw.text((12,28),'Evaluation only. No GT in control. GRAY = expired memory risk (action backup retained). Orange is not the future swept envelope.',fill='white')
    if replay_only:draw.text((12,45),'OFFLINE MEMORY REPLAY: images, ego motion and BLUE plan are recorded OLD driving. No new closed-loop execution.',fill=(255,210,100))
    ids=np.flatnonzero((scores>=.05)&np.isin(labels,[0,1,2,3,4])&np.isfinite(boxes).all(1)&(boxes[:,3:6]>=.2).all(1)&(boxes[:,3:6]<=30).all(1))
    for ci,name in enumerate(CAMERAS):
        rgb=np.asarray(packet['rgb'][name]);rgb=np.clip(rgb*255 if np.issubdtype(rgb.dtype,np.floating) and rgb.max()<=1.01 else rgb,0,255).astype('uint8')
        img=Image.fromarray(rgb).convert('RGB');d=ImageDraw.Draw(img)
        params=info['cam_params'][name]
        for j in ids:
            segments=projected_segments(box_corners(boxes[j]),params)
            for a,b in segments:d.line([tuple(a),tuple(b)],fill=RED,width=2)
            if segments:d.text(tuple(segments[0][0]),f'{j}:{scores[j]:.2f}',fill=RED)
        for a,b in projected_segments(target,params):d.line([tuple(a),tuple(b)],fill=GREEN,width=3)
        canvas.paste(img.resize((480,320)),(ci*480,60));draw.text((ci*480+8,65),name,fill='white')
    # Isotropic metric BEV, clipped separately so off-screen boxes cannot cover text.
    bev=Image.new('RGB',(280,322),(18,20,25));bd=ImageDraw.Draw(bev)
    def uv(points):
        p=np.asarray(points);return [(float(140+x*4.6),float(322-(y+5)*4.6)) for x,y in p[:,:2]]
    for y in range(0,61,10):
        p=uv([[0,y]])[0];bd.line([(0,p[1]),(280,p[1])],fill=(45,50,55));draw.text((18,p[1]+413),str(y)+'m',fill='white')
    def polygon(points,color,width=2):
        pts=uv(points);bd.line(pts+[pts[0]],fill=color,width=width)
    for j in ids:polygon(box_corners(boxes[j])[:4],RED)
    polygon(target[:4],GREEN,3)
    for row in shapes:
        b=np.asarray(row['box']);color=(140,150,165) if row['tier']=='history_risk' else ORANGE if row['tier']=='hard' else (180,140,80)
        if row.get('disk') and row.get('radius_m') is not None:
            theta=np.linspace(0,2*np.pi,65);r=row['radius_m']+row['margin_m'];xy=b[:2]+r*np.c_[np.cos(theta),np.sin(theta)]
        else:
            xy=box_corners(b)[:4]
        polygon(xy,color)
    selected=optimization.get('selected') or {};states=np.asarray(selected.get('states',[]))
    if len(states)>1:bd.line(uv(states[:,:2]),fill=(95,180,255),width=2)
    bd.text(uv([[0,0]])[0],'EGO',fill='white')
    canvas.paste(bev,(65,413));draw.rectangle((65,413,345,735),outline=(100,100,100))
    x=395;draw.text((x,410),'BEV: x right / y forward; blue = selected MPC plan',fill='white')
    draw.text((x,432),'Planner source: '+source,fill=ORANGE)
    draw.text((x,454),'Orange: base shapes (disk includes margin); box margins, heading/motion sweep, depth/occupancy omitted.',fill=ORANGE)
    for line,(key,title) in enumerate([('nearest_candidate','Candidate >= floor'),('nearest_retained_detection','Detection >= threshold')]):
        match=comparison[key]
        text=title+': none' if match is None else f'{title}: #{match["candidate_index"]}, score={match["score"]:.3f}, XY error={match["center_error_xy_m"]:.2f}m, speed={match["static_target_speed_error_mps"]:.2f}m/s, gate={match["within_proximity_gate"]}'
        draw.text((x,490+24*line),text,fill=RED)
    draw.text((x,550),'Nearest proximity does not confirm target identity; scores are not calibrated probabilities.',fill='white')
    draw.text((x,574),'Reference = inserted static asset Gaussian-support box, not certified physical geometry.',fill=GREEN)
    return canvas


def analyze(run,*,encoder=None):
    run=Path(run).resolve();out=run/'analysis/bbox_diagnostic';out.mkdir(parents=True,exist_ok=True)
    ref=_hugsim_expand(json.loads((run/'asset_reference.json').read_text()));report=_hugsim_expand(json.loads((run/'run_report.json').read_text()))
    if report.get('inserted_vehicle_static_world_pose') is not True:raise ValueError('Static target reference not declared')
    frames=out/'frames';frames.mkdir(exist_ok=True);rows=[];skipped=[];errors=[]
    folders=sorted((run/'uniad_worker').glob('step_*'))
    for folder in folders:
        index=int(folder.name.split('_')[-1])
        required=('perception.npz','observation.pkl','sensor_metadata.pkl','optimization.json','guard_memory_report.json')
        missing=[name for name in required if not (folder/name).exists()]
        if missing:skipped.append(dict(step=index,missing=missing));continue
        try:
            with (folder/'observation.pkl').open('rb') as f:packet,original_info=pickle.load(f)
            with (folder/'sensor_metadata.pkl').open('rb') as f:info=pickle.load(f)
            if float(info['timestamp'])!=float(original_info['timestamp']) or info['scene_token']!=original_info['scene_token']:raise ValueError('Sensor metadata mismatch')
            memory=_hugsim_expand(json.loads((folder/'guard_memory_report.json').read_text()));optimization=_hugsim_expand(json.loads((folder/'optimization.json').read_text()))
            with np.load(folder/'perception.npz',allow_pickle=False) as perception:
                if str(perception['box_convention'])!=CONVENTION:raise ValueError('Unknown detection convention')
                boxes=perception['candidate_boxes'];scores=perception['candidate_scores'];labels=perception['candidate_labels']
            target=target_corners(ref,info);comparison=compare(target,boxes,scores,labels,memory['detection_threshold'],memory['config']['candidate_floor'])
            visible=[name for name in CAMERAS if projected_segments(target,info['cam_params'][name])]
            shapes,source=planner_shapes(optimization,memory)
            row=dict(step=index,timestamp=float(info['timestamp']),source_frame=int(info['source_frame']),
                target_reference_corners_model=target.tolist(),in_camera_frustum=visible,occlusion_checked=False,
                target_velocity_world_mps=[0,0,0],planner_geometry_source=source,planner_proposals=shapes,
                expired_memory_risk_proposals=sum(s.get('expired_memory_risk_only',False) for s in shapes),
                retired_memory_count=len(memory.get('obstacle_memory',{}).get('lifecycle',{}).get('retired',[])),**comparison)
            rows.append(row)
            image=render_frame(packet,info,target,boxes,scores,labels,shapes,source,index,comparison,optimization,replay_only=report.get('replay_only',False))
            image.save(frames/f'{len(rows)-1:03d}.png')
        except Exception as exc:errors.append(dict(step=index,error=repr(exc)))
    retained=[r for r in rows if r['in_camera_frustum']]
    counts={key:sum(bool(r[key] and r[key]['within_proximity_gate']) for r in retained) for key in ('nearest_candidate','nearest_retained_detection')}
    result=dict(schema='vehicle_bbox_diagnostic_v1',passed=False,replay_only=report.get('replay_only',False),processed_frames=len(rows),
        camera_frustum_target_frames=len(retained),proximity_counts=counts,rows=rows,skipped_steps=skipped,errors=errors,
        ground_truth_used_by_controller=False,reference_only=True,physical_box_certified=False,
        association_identity_confirmed=False,recall_certified=False,occlusion_checked=False,
        planner_receipt_frames=sum(r['planner_geometry_source']=='actual_guard_arrays' for r in rows),
        planner_visualization_scope='proposal base shapes; disk includes margin; box margins, heading/motion sweeps, depth and occupancy omitted',
        expected_control_frames=len(report.get('trace',[])),
        missing_executed_control_steps=sorted(set(range(len(report.get('trace',[]))))-{r['step'] for r in rows}),
        reference_scope='single pinned static inserted vehicle only; other actors have detection overlays without reference matching',
        formal_evaluation_admitted=False)
    csvpath=out/'per_frame.csv'
    with csvpath.open('w',newline='') as f:
        fields=['step','timestamp','in_camera_frustum','candidate_index','candidate_score','candidate_xy_error_m','candidate_static_speed_error_mps','retained_index','retained_score','retained_xy_error_m','candidate_in_gate','retained_in_gate','planner_geometry_source']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            a=r['nearest_candidate'] or {};b=r['nearest_retained_detection'] or {}
            writer.writerow(dict(step=r['step'],timestamp=r['timestamp'],in_camera_frustum=bool(r['in_camera_frustum']),candidate_index=a.get('candidate_index'),candidate_score=a.get('score'),candidate_xy_error_m=a.get('center_error_xy_m'),candidate_static_speed_error_mps=a.get('static_target_speed_error_mps'),retained_index=b.get('candidate_index'),retained_score=b.get('score'),retained_xy_error_m=b.get('center_error_xy_m'),candidate_in_gate=a.get('within_proximity_gate'),retained_in_gate=b.get('within_proximity_gate'),planner_geometry_source=r['planner_geometry_source']))
    video=out/'vehicle_bbox_comparison.mp4'
    try:
        if not rows:raise RuntimeError('No complete diagnostic frames')
        if encoder is None:
            import imageio_ffmpeg
            encoder=imageio_ffmpeg.get_ffmpeg_exe()
        temporary=video.with_suffix('.encoding.mp4')
        with (out/'video.log').open('w') as log:
            subprocess.run([str(encoder),'-v','error','-y','-threads','1','-filter_threads','1','-framerate','10','-i',str(frames/'%03d.png'),'-frames:v',str(len(rows)),'-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(temporary)],stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
            os.replace(temporary,video)
            subprocess.run([str(encoder),'-v','error','-threads','1','-i',str(video),'-f','null','-'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
        result.update(video=str(video),video_decode_passed=True)
    except Exception as exc:result['video_error']=repr(exc)
    result['passed']=bool(rows) and not errors and result.get('video_decode_passed',False) and not result['missing_executed_control_steps']
    path=out/'report.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,path)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.run);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','skipped_steps')},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
