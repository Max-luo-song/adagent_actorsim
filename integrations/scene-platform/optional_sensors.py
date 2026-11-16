"""Read-only optional Gaussian depth scans and calibrated RGB-D point clouds."""

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

import copy,types,time
pass  # Path is provided by the portable boundary above
import numpy as np
from platform_io import asset,write,fingerprint
from sensor_contracts import configuration

FACE_DIRECTIONS=np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1],[0,0,-1]],float)

def face_rotation(direction):
    forward=np.asarray(direction,float);up=np.array([0.,0.,1.]) if abs(forward[2])<.9 else np.array([0.,1.,0.])
    right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
    return np.stack([right,down,forward],axis=1)

def beams(c):
    elev=np.deg2rad(np.linspace(c['elevation_min_deg'],c['elevation_max_deg'],c['channels']))
    az=np.linspace(-np.pi,np.pi,c['azimuth_samples'],endpoint=False)
    e,a=np.meshgrid(elev,az,indexing='ij')
    return np.stack([np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)],axis=-1).reshape(-1,3)

def sample_cube(views,c):
    """One organized range per configured ray; invalid returns stay zero with mask."""
    if len(views)!=6:raise ValueError('A complete six-face cube is required')
    rays=beams(c);face=np.argmax(rays@FACE_DIRECTIONS.T,axis=1);ranges=np.zeros(len(rays),np.float32);opacity=np.zeros_like(ranges)
    for index,view in enumerate(views):
        selected=np.flatnonzero(face==index);local=rays[selected]@face_rotation(FACE_DIRECTIONS[index]);k=np.asarray(view['K'])
        uv=local@k.T;uv=uv[:,:2]/uv[:,2,None];pixels=np.rint(uv).astype(int)
        depth=np.asarray(view['depth']);alpha=np.asarray(view['depth_alpha']);h,w=depth.shape
        inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        rows=selected[inside];x,y=pixels[inside].T;z=depth[y,x];a=alpha[y,x]
        distance=z/local[inside,2]
        okay=np.isfinite(distance)&np.isfinite(a)&(a>=c['minimum_opacity'])&(distance>=c['min_range_m'])&(distance<=c['max_range_m'])
        ranges[rows[okay]]=distance[okay];opacity[rows[okay]]=a[okay]
    valid=ranges>0
    return dict(xyz_sensor=(rays[valid]*ranges[valid,None]).astype(np.float32),range_m=ranges,
        valid=valid,opacity=opacity,ray_index=np.flatnonzero(valid).astype(np.int32),
        ring_index=(np.flatnonzero(valid)//c['azimuth_samples']).astype(np.int16),beam_directions=rays.astype(np.float32))

def rgbd_cloud(packet,c):
    sensor_world=np.asarray(packet['ego_to_world'])@np.asarray(c['sensor_to_ego']);world_sensor=np.linalg.inv(sensor_world)
    points=[];alphas=[];camera_indices=[]
    for i,name in enumerate(packet['camera_order']):
        view=packet['cameras'][name]
        if view.get('depth_encoding')!='expected_camera_z_v1':raise ValueError('Expected explicit camera-Z depth; no automatic guessing')
        depth=np.asarray(view['depth']);alpha=np.asarray(view['depth_alpha']);k=np.asarray(view['K'])
        if depth.shape!=alpha.shape or depth.ndim!=2:raise ValueError('Invalid depth/opacity dimensions')
        yy,xx=np.mgrid[0:depth.shape[0]:c['camera_pixel_stride'],0:depth.shape[1]:c['camera_pixel_stride']]
        z=depth[yy,xx].reshape(-1);a=alpha[yy,xx].reshape(-1)
        pixels=np.c_[xx.reshape(-1),yy.reshape(-1),np.ones(len(z))]
        xyz=pixels@np.linalg.inv(k).T*z[:,None];transform=world_sensor@np.asarray(view['camera_to_world'])
        xyz=xyz@transform[:3,:3].T+transform[:3,3];distance=np.linalg.norm(xyz,axis=1)
        good=np.isfinite(xyz).all(1)&np.isfinite(a)&(z>0)&(a>=c['minimum_opacity'])&(distance>=c['min_range_m'])&(distance<=c['max_range_m'])
        points.append(xyz[good]);alphas.append(a[good]);camera_indices.append(np.full(good.sum(),i,np.int16))
    return dict(xyz_sensor=np.concatenate(points).astype(np.float32),opacity=np.concatenate(alphas).astype(np.float32),camera_index=np.concatenate(camera_indices))

def render_cube(backend,pose,timestamp,c):
    """Reuse native backend and actor render_context. Never load another model."""
    torch=backend.torch;n=c['cube_resolution'];mount=np.asarray(c['sensor_to_ego']);views=[]
    original=backend.cameras_data;last=backend._last
    template=next(iter(original.values()));k=np.array([[n/2,0,(n-1)/2],[0,n/2,(n-1)/2],[0,0,1]],np.float32)
    try:
        for start in [0,3]:
            cameras={}
            for offset,(key,old) in enumerate(sorted(original.items())):
                if offset>=3:raise ValueError('Registered backend requires exactly three camera slots')
                camera_pose=np.eye(4);camera_pose[:3,:3]=face_rotation(FACE_DIRECTIONS[start+offset])
                cameras[key]=types.SimpleNamespace(cam_to_ego=mount@camera_pose,is_fisheye=False,
                    intrinsics=torch.as_tensor(k,device=template.intrinsics.device).unsqueeze(0).repeat(backend.frames,1,1),
                    distortions=old.distortions,unique_img_idx=old.unique_img_idx,HEIGHT=n,WIDTH=n,egocar_mask=None,ego_template=None)
            if len(cameras)!=3:raise ValueError('Expected three native camera slots')
            backend.cameras_data=cameras;raw=backend.render(pose,timestamp)
            if len(raw['cameras'])!=3:raise ValueError('Missing cube faces')
            for name in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT']:
                view=raw['cameras'][name]
                if view.get('depth_encoding')!='expected_camera_z_v1':raise ValueError('Wrong depth semantics')
                views.append(view)
    finally:backend.cameras_data=original;backend._last=last
    return views

class Recorder:
    def __init__(self,value,out,event=lambda *args:None):
        self.config=configuration(value);self.out=Path(out);self.event=event;self.frames=[]
    def record(self,session,packet):
        c=self.config
        if not c['enabled']:return None
        if not hasattr(self,'start_frame'):self.start_frame=packet['source_frame']
        index=int(packet['source_frame'])-self.start_frame
        if index%c['record_every_n_steps']:return None
        began=time.monotonic();folder=self.out/f'{index:04d}'
        if folder.exists():raise FileExistsError('Sensor frame already exists; no stale reuse')
        folder.mkdir(parents=True)
        source=float(packet['timestamp_s']);pose=np.asarray(packet['ego_to_world']);mount=np.asarray(c['sensor_to_ego'])
        if not np.isclose(source*session.spec.fps,packet['source_frame'],atol=1e-6):raise ValueError('Sensor source clock mismatch')
        if c['provider']=='camera_depth_pointcloud_v1':data=rgbd_cloud(packet,c);renders=0
        else:data=sample_cube(render_cube(session.backend,pose,source,c),c);renders=6
        data['sensor_to_ego']=mount;data['sensor_to_world']=pose@mount
        data['source_timestamp_s']=np.array(source);data['source_frame']=np.array(packet['source_frame'])
        data['local_timestamp_s']=np.array(index/session.spec.fps)
        path=folder/'points.npz';np.savez_compressed(path,**data)
        ply=folder/'points.ply'
        with ply.open('wb') as stream:
            stream.write(('ply\nformat binary_little_endian 1.0\ncomment diagnostic Gaussian/depth simulation; no physical lidar certification\nelement vertex '+str(len(data['xyz_sensor']))+'\nproperty float x\nproperty float y\nproperty float z\nend_header\n').encode())
            stream.write(data['xyz_sensor'].astype('<f4').tobytes())
        from PIL import Image,ImageDraw
        preview=Image.new('RGB',(640,640),'#181818');draw=ImageDraw.Draw(preview)
        xyz=data['xyz_sensor'];extent=min(c['max_range_m'],50.);scale=290/extent
        keep=(abs(xyz[:,0])<extent)&(abs(xyz[:,1])<extent)
        for x,y in xyz[keep][::max(1,len(xyz)//20000),:2]:
            px,py=int(320-y*scale),int(320-x*scale)
            if 0<=px<640 and 0<=py<640:draw.point((px,py),fill='#69d9ef')
        draw.text((8,8),f'{c["provider"]} | frame {packet["source_frame"]} | {len(xyz)} points',fill='white')
        draw.text((8,26),'x forward/up; y left | diagnostic; not physical lidar',fill='white')
        image=folder/'pointcloud_bev.png';preview.save(image)
        # Diagnostic evidence only: both rendering paths share the edited model tensors.
        models=getattr(session.backend,'env',None);shared={}
        if models is not None:
            for name,state in models.models.items():
                model=session.backend.trainer.models.get(name)
                if model is not None:shared[name]=state['instances_trans'] is model.instances_trans
        record=dict(schema='optional_pointcloud_frame_v1',provider=c['provider'],source_frame=packet['source_frame'],
            source_timestamp_s=source,local_timestamp_s=index/session.spec.fps,sensor_to_ego=mount.tolist(),sensor_to_world=(pose@mount).tolist(),
            frame_convention='right-handed x-forward y-left z-up; points in configured sensor frame; metres',
            generation=session.generation,revision=session.revision,point_count=len(data['xyz_sensor']),
            scene_state_token=packet['model_cache_token'],shared_actor_tensors=shared,
            source='current edited Gaussian scene' if renders else 'current calibrated camera depth',
            extra_gaussian_view_renders=renders,physical_lidar_certified=False,feeds_UniAD=False,feeds_controller=False,
            intensity_available=False,doppler_available=False,rolling_scan=False,
            geometry_is_expected_depth_not_certified_first_return=True,points=asset(path),ply=asset(ply),preview=asset(image),wall_seconds=time.monotonic()-began)
        write(folder/'receipt.json',record);self.frames.append(asset(folder/'receipt.json'));self.flush()
        return record
    def flush(self):
        if not self.config['enabled']:return
        write(self.out/'manifest.json',dict(schema='optional_pointcloud_sequence_v1',config=self.config,frames=self.frames,
            frame_count=len(self.frames),feeds_UniAD=False,feeds_controller=False,physical_lidar_certified=False))

def summarize(task,work,report):
    from platform_io import read,verify
    c=configuration(task.get('sensors',{}).get('lidar'));path=Path(work)/'optional_sensors/manifest.json'
    result=dict(enabled=c['enabled'],provider=c['provider'],feeds_UniAD=False,feeds_controller=False,
        physical_lidar_certified=False,frame_count=0,checks_passed=True,artifacts=[])
    if not c['enabled']:return result
    result['checks_passed']=False
    if not path.exists():result['error']='Enabled recorder has no manifest';return result
    manifest=read(path);rows=[]
    for a in manifest['frames']:
        verify(a);row=read(a['path']);rows.append(row)
        for key in ['points','ply','preview']:verify(row[key])
    expected=[o['index'] for o in report['observations'] if o['index']%c['record_every_n_steps']==0]
    start=report['initial_context']['source_frame']
    checks=dict(configuration_matches=manifest['config']==c,
        frame_clock=[r['source_frame'] for r in rows]==[i+start for i in expected],
        source_time=all(np.isclose(r['source_timestamp_s'],r['source_frame']*.1) for r in rows),
        edited_revision=all(r['revision']==(1 if task['edits'] else 0) for r in rows),
        shared_actor_tensors=all(r['shared_actor_tensors'] and all(r['shared_actor_tensors'].values()) for r in rows),
        same_generation=len({r['generation'] for r in rows})==1,
        no_online_consumer=all(r['feeds_UniAD'] is False and r['feeds_controller'] is False for r in rows))
    result.update(frame_count=len(rows),point_counts=[r['point_count'] for r in rows],checks=checks,checks_passed=bool(rows) and all(checks.values()),
        extra_gaussian_view_renders=sum(r['extra_gaussian_view_renders'] for r in rows),
        artifacts=[dict(kind='optional_sensor_sequence',**asset(path))])
    return result
