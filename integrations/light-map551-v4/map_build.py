"""Offline, explicitly authorized static scene map from original calibrated sensors.

No map truth is invented: the right boundary is the observed solid white edge;
the opposite limit is a conservative connected-road inner envelope (the left
side of this carriageway has a median/curb, not a fabricated white line).
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

import sys
pass  # Path is provided by the portable boundary above
import numpy as np
from scipy.ndimage import binary_closing,median_filter
from common import *

RAW_PARENT=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/data/waymo/raw'))
TRAIN_LIST=Path(_hugsim_expand('${HUGSIM_EXTERNAL_ROOT}/adscenegraph/Omniscenegraph/data/waymo_train_list.txt'))

def load_frame(frame,source):
    import cv2
    from uniad_bridge import to_uniad,ORDER
    from build_source_sequence import training_lidar_mount
    raw=Path(source['raw_data']);origin=np.loadtxt(raw/'ego_pose/000.txt')
    pose=np.linalg.inv(origin)@np.loadtxt(raw/f'ego_pose/{frame:03d}.txt');cams={};masks={};assets=[]
    for cid,name in [('0','CAM_FRONT'),('1','CAM_FRONT_LEFT'),('2','CAM_FRONT_RIGHT')]:
        c=source['cameras'][cid];image=raw/f'images/{frame:03d}_{cid}.jpg';road=raw/f'road_masks/{frame:03d}_{cid}.png'
        rgb=cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(road),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=rgb.shape[:2]:raise ValueError('Road mask/image mismatch')
        h,w=mask.shape;fx,fy,cx,cy,*dist=c['intrinsics_9'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        mx,my=cv2.initUndistortRectifyMap(K,np.array(dist),None,K,(w,h),cv2.CV_32FC1)
        rgb=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR);masks[name]=cv2.remap(mask,mx,my,cv2.INTER_NEAREST)>127
        T=np.eye(4);T[:3,:3]=c['opencv_to_waymo'];T=np.asarray(c['camera_waymo_to_ego'])@T
        cams[name]=dict(rgb=rgb,K=K,camera_to_ego=T,camera_to_world=pose@T,depth=np.zeros((h,w),np.float32))
        assets.extend([asset(image),asset(road)])
    mount=training_lidar_mount(ROOT/'code/HUGSIM/configs/sim/nuscenes_camera.yaml')
    packet=dict(schema='unified_observation_v1',camera_order=ORDER,training=False,formal_evaluation_admitted=False,
        ego_to_world=pose,timestamp_s=frame*.1,source_frame=frame,edit_trust=dict(edited=True),cameras=cams)
    obs,info=to_uniad(packet,mount,speed_mps=0.,steering_rad=0.,command=2,scene_token='static_map_build_offline')
    assets.append(asset(raw/f'ego_pose/{frame:03d}.txt'))
    return pose,cams,masks,obs,info,assets

def build(folder):
    configure();import cv2
    sys.path.insert(0,str(CR/'phase46/framework_driver_v1'))
    from white_edge import detect
    source=read(BUNDLE/'source_manifest.json');folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    frames=list(range(0,199,10))+[198];right_points=[];left_samples=[];receipts=[];used=[]
    # Offline views have overlapping longitudinal support. No online image
    # detections are needed after this asset is frozen.
    forward=np.arange(8.,40.01,.5);left=np.arange(0.,18.01,.1)
    F,L=np.meshgrid(forward,left,indexing='ij');local=np.c_[F.ravel(),L.ravel(),np.zeros(F.size)]
    for frame in frames:
        pose,cams,masks,obs,info,assets=load_frame(frame,source);used.extend(assets);packet=detect(obs,info)
        world=local@pose[:3,:3].T+pose[:3,3];road_votes=np.zeros(len(local));seen=np.zeros(len(local))
        for name,cam in cams.items():
            inv=np.linalg.inv(cam['camera_to_ego']);p=local@inv[:3,:3].T+inv[:3,3];q=p@cam['K'].T
            uv=q[:,:2]/np.maximum(q[:,2,None],1e-10);h,w=masks[name].shape
            valid=(p[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
            pix=np.rint(np.clip(uv,[0,0],[w-1,h-1])).astype(int)
            road_votes+=valid*masks[name][pix[:,1],pix[:,0]];seen+=valid
        mask=(road_votes>=np.maximum(seen*.5,1)).reshape(F.shape);valid=(seen>0).reshape(F.shape)
        # Close small object/segmentation holes; a sustained non-road strip
        # still bounds the connected carriageway. Mark this as a conservative
        # segmentation-derived inner boundary, not a white-line annotation.
        for j in range(len(forward)):
            row=mask[j];known=valid[j]
            if not known[0] or not row[0]:continue
            row=binary_closing(np.pad(row,(0,15),constant_values=False),structure=np.ones(11,bool))[:len(left)]
            # scipy closing erodes raster ends; restore only original near-ego support.
            row[:5]=mask[j,:5]
            stop=None
            for k in range(5,len(left)-8):
                if known[k:k+8].all() and not row[k:k+8].any():stop=k;break
            if stop is None:continue
            point=world.reshape(F.shape+(3,))[j,stop]
            left_samples.append([*point[:2],frame])
        accepted=[]
        for edge in packet['edges']:
            if edge['side']!='right':continue
            ys=np.arange(edge['observed_y_range'][0],edge['observed_y_range'][1]+.01,.5)
            points=np.c_[edge['a']*ys+edge['b'],ys,np.full(len(ys),edge['plane_height_m'])]
            virtual=points@np.asarray(info['explicit_l2g_r_mat']).T+info['explicit_l2g_t']
            points=np.c_[virtual[:,1],-virtual[:,0],virtual[:,2]]
            if np.median(points[:,1])>=pose[1,3]:continue
            right_points.extend(np.c_[points[:,:2],np.full(len(points),frame)].tolist());accepted.append(edge)
        receipts.append(dict(frame=frame,right_edges=accepted,left_samples_so_far=len(left_samples)))
        print('Offline static-map source frame',frame,'right points',len(right_points),'left samples',len(left_samples),flush=True)
    rp=np.asarray(right_points);lp=np.asarray(left_samples)
    if len(rp)<200 or len(lp)<200:raise ValueError('Insufficient observed boundary support; no guessed road width admitted')
    def robust_line(points):
        good=np.ones(len(points),bool)
        for _ in range(6):
            fit=np.polyfit(points[good,0],points[good,1],1);err=points[:,1]-np.polyval(fit,points[:,0]);median=np.median(err[good])
            scale=max(.08,1.4826*np.median(abs(err[good]-median)));good=abs(err-median)<3*scale
        return fit,good,err
    rf,rg,re=robust_line(rp)
    # Avoid treating occasional occluding objects/opposite carriageway as the
    # outer boundary. Median per 10m station, then take a conservative lower
    # envelope and record its support and uncertainty.
    bins=np.arange(10.,411.,10.);med=[]
    for x in bins:
        p=lp[(abs(lp[:,0]-x)<7.5)&(lp[:,1]>1.)]
        if len(p)>=5:med.append([x,float(np.quantile(p[:,1],.25))])
    med=np.asarray(med)
    if len(med)<20:raise ValueError('Left road support insufficient; do not fabricate a boundary')
    lf,lg,le=robust_line(med)
    # Retain a conservative connected-road inner envelope, with uncertainty
    # disclosed independently of the observed right white line.
    left_offset=float(np.quantile(med[lg,1]-np.polyval(lf,med[lg,0]),.1))-.25
    lf[-1]+=left_offset
    xmin=max(float(rp[rg,0].min()),float(med[lg,0].min()));xmax=min(float(rp[rg,0].max()),float(med[lg,0].max()))
    xs=np.arange(np.ceil(xmin),np.floor(xmax)+.01,1.);right=np.polyval(rf,xs);left_limit=np.polyval(lf,xs)
    np.savez_compressed(folder/'support.npz',right_observations=rp,left_road_samples=lp,left_station_envelope=med)
    write(folder/'fit_diagnostic.json',dict(right_fit=rf.tolist(),left_fit=lf.tolist(),xmin=xmin,xmax=xmax,
        stations=len(xs),minimum_width_m=float(np.min(left_limit-right)),right_fit_inliers=int(rg.sum()),left_fit_inliers=int(lg.sum())))
    if len(xs)<350 or np.min(left_limit-right)<3.:raise ValueError('Map extent/connected-road width failed; inspect fit_diagnostic.json')
    if np.quantile(abs(re[rg]),.95)>.25:raise ValueError('Road is not adequately represented by the fitted static boundary')
    vertices=np.vstack([np.c_[xs,right],np.c_[xs[::-1],left_limit[::-1]]])
    np.savez_compressed(folder/'support.npz',right_observations=rp,left_road_samples=lp,left_station_envelope=med)
    # Match actual source identity; the available scene-flow record has no field10 map.
    segment=TRAIN_LIST.read_text().splitlines()[551];raw_record=RAW_PARENT/(segment+'.tfrecord')
    mapping=dict(schema='authorized_static_scene_map_v1',scene=551,coordinate_system='first_waymo_ego_forward_left_up_meters',
        vertices=vertices.tolist(),longitudinal_domain_m=[float(xs[0]),float(xs[-1])],
        right_line_coefficients=rf.tolist(),left_inner_coefficients=lf.tolist(),uncertainty_m=.30,
        right_boundary='offline fitted observed solid white road-edge line',
        left_boundary='conservative connected-road segmentation inner envelope; median side, not a fabricated white line',
        source_kind='offline calibrated sensor map, no independent HD survey',
        authorization='User explicitly permits controller to use the static map and requests area inside road boundaries; no per-frame line detection',
        online_static_map_allowed=True,online_actor_truth_allowed=False,online_future_ego_trajectory_allowed=False,
        source_segment=segment,raw_record=str(raw_record),raw_HD_map_available=False,
        source_manifest=asset(BUNDLE/'source_manifest.json'),sources=used,support=asset(folder/'support.npz'),
        offline_frames=frames,right_fit_95pct_residual_m=float(np.quantile(abs(re[rg]),.95)),
        left_fit_95pct_residual_m=float(np.quantile(abs(le[lg]),.95)),
        hard_limits='full footprint plus uncertainty and actuator-aware braking; map exterior is not drivable')
    write(folder/'map.json',mapping);write(folder/'construction.json',dict(receipts=receipts,online_detections_required=0,
        raw_schema_reference='https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto',
        map_type_reference='https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto'))
    plot(folder,mapping,rp,lp,med)
    return mapping

def plot(folder,mapping,rp,lp,med):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,4));p=np.asarray(mapping['vertices']);ax.fill(p[:,0],p[:,1],alpha=.2,label='authorized drivable polygon')
    ax.scatter(rp[:,0],rp[:,1],s=2,label='observed right solid-white support');ax.scatter(lp[:,0],lp[:,1],s=1,alpha=.15,label='connected-road mask boundary samples')
    ax.plot(med[:,0],med[:,1],'.',label='conservative station envelope');ax.set(xlabel='Source forward x (m)',ylabel='Source left y (m)',ylim=(-4,18));ax.legend();fig.tight_layout();fig.savefig(Path(folder)/'map_review.png',dpi=160);plt.close(fig)

def fuse_static_masks(folder):
    """Recover road hidden by moving cars using side views and other source times.

    Reuse already computed right-line observations. Unlike the rejected first
    fit, moving-object holes are fused in world coordinates before choosing
    the connected-road boundary. No runtime detections are introduced.
    """
    configure();import cv2
    folder=Path(folder)
    if (folder/'map.json').exists():raise FileExistsError('A completed map already exists')
    source=read(BUNDLE/'source_manifest.json');raw=Path(source['raw_data']);origin=np.loadtxt(raw/'ego_pose/000.txt')
    with np.load(folder/'support.npz') as z:rp=z['right_observations'].copy()
    right_support=asset(folder/'support.npz');diag=read(folder/'fit_diagnostic.json');rf=np.asarray(diag['right_fit'])
    xs=np.arange(8.,429.,.5);ys=np.arange(0.,18.01,.1);X,Y=np.meshgrid(xs,ys,indexing='ij')
    yes=np.zeros(X.shape,np.uint16);seen=np.zeros(X.shape,np.uint16);sources=[];frames=list(range(0,199,5))+[198]
    for frame in frames:
        posefile=raw/f'ego_pose/{frame:03d}.txt';pose=np.linalg.inv(origin)@np.loadtxt(posefile);sources.append(asset(posefile))
        # Ground plane is local to each recorded pose, not a flat whole-scene Z.
        z=pose[2,3]-(pose[0,2]*(X-pose[0,3])+pose[1,2]*(Y-pose[1,3]))/pose[2,2]
        points=np.c_[X.ravel(),Y.ravel(),z.ravel()];local=(points-pose[:3,3])@pose[:3,:3]
        near=(local[:,0]>=-15)&(local[:,0]<=40)&(abs(local[:,1])<=20)
        votes=np.zeros(len(points));views=np.zeros(len(points))
        for cid in ('0','1','2','3','4'):
            name=raw/f'road_masks/{frame:03d}_{cid}.png';mask=cv2.imread(str(name),cv2.IMREAD_GRAYSCALE);sources.append(asset(name))
            c=source['cameras'][cid];fx,fy,cx,cy,*dist=c['intrinsics_9'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            # Project into the actual distorted image coordinates of the mask.
            T=np.eye(4);T[:3,:3]=c['opencv_to_waymo'];T=np.asarray(c['camera_waymo_to_ego'])@T;inv=np.linalg.inv(T)
            p=local@inv[:3,:3].T+inv[:3,3];ids=np.flatnonzero(near&(p[:,2]>.1))
            if not len(ids):continue
            uv,_=cv2.projectPoints(p[ids],np.zeros(3),np.zeros(3),K,np.asarray(dist));uv=uv[:,0]
            h,w=mask.shape;valid=(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
            ids=ids[valid];uv=np.rint(uv[valid]).astype(int);votes[ids]+=mask[uv[:,1],uv[:,0]]>127;views[ids]+=1
        valid=views>0;road=votes>=np.maximum(1,views*.5)
        seen+=valid.reshape(X.shape);yes+=(road&valid).reshape(X.shape)
        if frame%20==0:print('World-fused original road-mask frame',frame,flush=True)
    road=(yes>=np.maximum(1,seen*.5));boundary=[]
    for i,x in enumerate(xs):
        if not seen[i,0] or not road[i,0]:continue
        row=binary_closing(np.pad(road[i],(8,8),constant_values=True),structure=np.ones(9,bool))[8:-8]
        for j in range(5,len(ys)-8):
            if (seen[i,j:j+8]>0).all() and not row[j:j+8].any():
                boundary.append([x,ys[j]]);break
    boundary=np.asarray(boundary)
    # Retain measured local shape instead of fitting a spurious declining line
    # through moving-car occlusion tracks. A conservative smoothed envelope is
    # inside the identified connected road, with no constant lane-width guess.
    stations=np.arange(15.,416.,5.);med=[]
    for x in stations:
        p=boundary[abs(boundary[:,0]-x)<=3]
        if len(p)>=4:med.append([x,float(np.median(p[:,1]))])
    med=np.asarray(med);np.savez_compressed(folder/'fused_masks.npz',xs=xs,ys=ys,road_votes=yes,observed_votes=seen,boundary=boundary,station_boundary=med)
    write(folder/'fusion_sources.json',dict(sources=sources,right_support_reused=right_support,frames=frames))
    return finish_fused_map(folder)

def finish_fused_map(folder):
    folder=Path(folder);source=read(BUNDLE/'source_manifest.json')
    with np.load(folder/'fused_masks.npz') as z:
        xs=z['xs'];ys=z['ys'];yes=z['road_votes'];seen=z['observed_votes'];boundary=z['boundary'];med=z['station_boundary']
    with np.load(folder/'support.npz') as z:rp=z['right_observations']
    rf=np.asarray(read(folder/'fit_diagnostic.json')['right_fit']);road=yes>=np.maximum(1,seen*.5)
    # Cross streets interrupt the median: do not expand the authorized corridor
    # into a whole intersection. Bridge its inner envelope only when the
    # original multi-view mask actually observes pavement along that bridge.
    bridges=[];filled=[]
    for j in range(len(med)-1):
        filled.append(med[j].tolist());a,b=med[j:j+2]
        if b[0]-a[0]<=5.01:continue
        if b[0]-a[0]>50:raise ValueError('Unbounded map gap; no unobserved bridge allowed')
        station=np.arange(a[0]+5,b[0],5.);new=[]
        for x in station:
            y=float(np.interp(x,[a[0],b[0]],[a[1],b[1]]))-.3
            ix=abs(xs-x).argmin();iy=abs(ys-y).argmin()
            if not (seen[ix,iy]>0 and road[ix,iy]):raise ValueError('Proposed conservative intersection bridge lacks observed road support')
            new.append([float(x),y])
        filled.extend(new);bridges.append(dict(from_x=float(a[0]),to_x=float(b[0]),type='conservative inner corridor across observed open pavement',stations=new))
    filled.append(med[-1].tolist());med=np.asarray(filled)
    if len(med)<65 or np.diff(med[:,0]).max()>5.01:raise ValueError('World-fused static road coverage still insufficient')
    clean=median_filter(med[:,1],size=5,mode='nearest')-.30
    xx=np.arange(max(15.,float(rp[:,0].min())),min(float(med[-1,0]),float(rp[:,0].max()))+.01,1.)
    right=np.polyval(rf,xx);left=np.interp(xx,med[:,0],clean)
    if np.min(left-right)<3.:raise ValueError('Connected-road envelope too narrow; do not relax geometry to force admission')
    vertices=np.vstack([np.c_[xx,right],np.c_[xx[::-1],left[::-1]]])
    segment=TRAIN_LIST.read_text().splitlines()[551]
    mapping=dict(schema='authorized_static_scene_map_v1',scene=551,coordinate_system='first_waymo_ego_forward_left_up_meters',
        vertices=vertices.tolist(),longitudinal_domain_m=[float(xx[0]),float(xx[-1])],right_line_coefficients=rf.tolist(),
        left_stations=np.c_[med[:,0],clean].tolist(),uncertainty_m=.30,
        right_boundary='offline fitted observed solid white road-edge line',left_boundary='world-fused connected-road conservative inner envelope, median side; not an asserted white line',
        source_kind='offline calibrated original sensor map; not an independent HD survey',
        authorization='User explicitly permits controller to use map inside road boundaries without per-frame white-line detection',
        online_static_map_allowed=True,online_actor_truth_allowed=False,online_future_ego_trajectory_allowed=False,
        source_segment=segment,raw_HD_map_available=False,source_manifest=asset(BUNDLE/'source_manifest.json'),
        support=asset(folder/'support.npz'),fusion_support=asset(folder/'fused_masks.npz'),source_receipts=asset(folder/'fusion_sources.json'),
        offline_frames=read(folder/'fusion_sources.json')['frames'],intersection_inner_bridges=bridges,
        hard_limits='full footprint plus map uncertainty, actuator-aware braking; map exterior is non-drivable')
    write(folder/'map.json',mapping)
    plot(folder,mapping,rp,boundary,med)
    return mapping

def validate_overlay(folder):
    """Independent projection review along the parts where Hough lost the edge."""
    configure();import cv2
    from PIL import Image,ImageDraw
    folder=Path(folder);mapping=read(folder/'map.json');source=read(BUNDLE/'source_manifest.json');raw=Path(source['raw_data'])
    origin=np.loadtxt(raw/'ego_pose/000.txt');rows=[]
    for frame in (0,50,100,130,150,170,190,198):
        pose=np.linalg.inv(origin)@np.loadtxt(raw/f'ego_pose/{frame:03d}.txt')
        xx=pose[0,3]+np.arange(10.,30.01,.25);xx=xx[(xx>=mapping['longitudinal_domain_m'][0])&(xx<=mapping['longitudinal_domain_m'][1])]
        if len(xx)<10:continue
        left_stations=np.asarray(mapping['left_stations']);right=np.polyval(mapping['right_line_coefficients'],xx);left=np.interp(xx,left_stations[:,0],left_stations[:,1])
        for cid,boundary,label in [('0',right,'right solid-white map'),('1',left,'conservative left road limit')]:
            c=source['cameras'][cid];fx,fy,cx,cy,*dist=c['intrinsics_9'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            T=np.eye(4);T[:3,:3]=c['opencv_to_waymo'];T=np.asarray(c['camera_waymo_to_ego'])@T
            def project(y):
                z=pose[2,3]-(pose[0,2]*(xx-pose[0,3])+pose[1,2]*(y-pose[1,3]))/pose[2,2]
                local=(np.c_[xx,y,z]-pose[:3,3])@pose[:3,:3];inv=np.linalg.inv(T);p=local@inv[:3,:3].T+inv[:3,3]
                uv,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),K,np.array(dist));return uv[:,0],p[:,2]
            rgb=cv2.cvtColor(cv2.imread(str(raw/f'images/{frame:03d}_{cid}.jpg')),cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
            uv,depth=project(boundary);valid=(depth>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
            draw.line([tuple(p) for p in uv[valid]],fill='red' if cid=='0' else 'magenta',width=4);draw.text((20,20),f'source frame {frame}: {label}; offline only',fill='red',stroke_width=1)
            image.save(folder/f'projection_{frame:03d}_{cid}.jpg')
            if cid!='0':continue
            lum=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);contrast=lum.astype(float)-cv2.medianBlur(lum,25).astype(float)
            white=(rgb.max(2).astype(float)-rgb.min(2)<45)&(lum>110)&(contrast>15)
            hit=np.zeros(len(xx),bool)
            for offset in np.arange(-.30,.301,.025):
                q,d=project(boundary+offset);v=(d>0)&(q[:,0]>=0)&(q[:,0]<w-1)&(q[:,1]>=0)&(q[:,1]<h-1)
                p=np.rint(np.clip(q,[0,0],[w-1,h-1])).astype(int);hit|=v&white[p[:,1],p[:,0]]
            rows.append(dict(frame=frame,projected_points=int(valid.sum()),white_pixel_support_within_03m=float(hit[valid].mean()) if valid.any() else 0.))
    report=dict(rows=rows,source='offline raw image/calibration validation, no online detector dependency',
        every_sample_has_white_support=all(r['white_pixel_support_within_03m']>=.5 for r in rows),independent_HD_truth=False)
    write(folder/'projection_review.json',report);return report
