"""Geometry-consistency lifecycle for unobserved tracking cache entries.

Read-only observer: tags unsupported, motion-inconsistent decoded geometry.
It never touches neural state, scores, BEV, RNG or driver obstacle memory.
No scene numbers, track IDs, evaluation truth or fixed intervention frames.
"""
import numpy as np
from box_geometry import box_rotation_row
from depth_contract import geometry_depth,reliable_mask
from input_geometry import intrinsic_matrix

class EvidenceHistory:
    def __init__(self,config):
        self.config=dict(config);self.scene=None;self.time=None;self.records={}
    def observe(self,boxes,ids,labels,obs,info):
        boxes=np.asarray(boxes,float);ids=np.asarray(ids);labels=np.asarray(labels)
        if boxes.ndim!=2 or boxes.shape[1]!=9 or len(ids)!=len(boxes) or len(labels)!=len(boxes):raise ValueError('Unaligned lifecycle inputs')
        now=float(info['timestamp']);scene=info['scene_token'];cfg=self.config
        if not np.isfinite(now):raise ValueError('Invalid time')
        reset=self.time is not None and (scene!=self.scene or now<=self.time or now-self.time>cfg['max_gap_seconds'])
        if reset:self.records={}
        self.scene=scene;self.time=now
        rotation=np.asarray(info['explicit_l2g_r_mat'],float);translation=np.asarray(info['explicit_l2g_t'],float)
        if rotation.shape!=(3,3) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():raise ValueError('Invalid pose')
        cameras=[]
        if info.get('depth_encoding')=='expected_camera_z_v1':
            for name in obs.get('depth',{}):
                camera=info['cam_params'][name]
                cameras.append((np.asarray(camera['l2c'],float),intrinsic_matrix(camera['intrinsic'])[:3,:3],geometry_depth(obs,name),reliable_mask(obs,name)))
        active_ids=ids[ids>=0]
        if len(set(map(int,active_ids)))!=len(active_ids):raise ValueError('Duplicate neural IDs in one frame')
        rows=[];margin=float(cfg['obstacle_margin_m'])
        for index,(box,identity,label) in enumerate(zip(boxes,ids,labels)):
            identity=int(identity)
            if identity<0:continue
            if not np.isfinite(box).all() or (box[3:6]<=0).any():continue
            center=box[:3].copy();center[2]+=box[5]/2
            visible_center=False;support=occluded=observed=0
            # Interior sampling avoids interpreting a projected corner as a
            # full observation. Any positive support/occlusion vetoes retirement.
            x,y,z=np.meshgrid(np.linspace(-box[3]/2,box[3]/2,5),np.linspace(-box[4]/2,box[4]/2,9),np.linspace(.2*box[5],.8*box[5],3),indexing='ij')
            xy=np.c_[x.ravel(),y.ravel()]@box_rotation_row(box[6])+box[:2]
            points=np.vstack([center,np.c_[xy,z.ravel()+box[2]]])
            for transform,k,depth,valid in cameras:
                cp=(transform@np.c_[points,np.ones(len(points))].T).T[:,:3];uv=cp@k.T;uv=uv[:,:2]/np.maximum(cp[:,2,None],1e-8)
                inside=(cp[:,2]>.1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0])
                visible_center|=bool(inside[0]);ix=np.flatnonzero(inside[1:])+1
                pixels=uv[ix].astype(int)
                if not len(ix):continue
                good=valid[pixels[:,1],pixels[:,0]];ix=ix[good];pixels=pixels[good]
                if not len(ix):continue
                # Count independent pixels, rather than duplicate volume samples.
                _,unique=np.unique(pixels,axis=0,return_index=True);ix=ix[unique];pixels=pixels[unique]
                gap=depth[pixels[:,1],pixels[:,0]]-cp[ix,2]
                observed+=len(gap);support+=int((np.abs(gap)<=margin).sum());occluded+=int((gap < -margin).sum())
            world=rotation@box[:3]+translation;velocity=rotation@np.r_[box[7:9],0.]
            previous=self.records.get(identity,dict(history=[],ever_observed=False))
            ever=previous['ever_observed'] or visible_center or support>0
            history=[h for h in previous['history'] if now-h[0]<=cfg['max_gap_seconds']]
            history.append((now,world.copy(),velocity.copy()))
            self.records[identity]=dict(history=history,ever_observed=ever,last_seen=now)
            rows.append(dict(index=index,identity=identity,label=int(label),box=box,visible_center=visible_center,support_pixels=support,occluded_pixels=occluded,reliable_pixels=observed,ever_observed=ever,history=history))
        # Same-frame duplicates can share observation protection, never create
        # extra temporal observations or confirmation for each other.
        parent=list(range(len(rows)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for i,a in enumerate(rows):
            for j in range(i):
                b=rows[j]
                ratio=a['box'][3:6]/b['box'][3:6]
                if a['label']==b['label'] and (.5<=ratio).all() and (ratio<=2).all() and np.linalg.norm(a['box'][:2]-b['box'][:2])<min(1.,min(a['box'][3:5])/2):parent[find(i)]=find(j)
        groups={}
        for i in range(len(rows)):groups.setdefault(find(i),[]).append(i)
        retire=[];audits=[]
        for indices in groups.values():
            protected=any(rows[i]['ever_observed'] or rows[i]['occluded_pixels'] for i in indices)
            for i in indices:
                row=rows[i];h=row['history'];record=self.records[row['identity']]
                if protected:record['ever_observed']=True
                enough=len(h)>=cfg['confirmation_observations'] and now-h[0][0]>=cfg['confirmation_seconds']-1e-6
                # Integrated model velocity must explain decoded world motion
                # within the SAME metric error budget used by the planner.
                innovation=0.
                predicted=h[0][1].copy()
                for a,b in zip(h,h[1:]):
                    predicted+=(a[2]+b[2])/2*(b[0]-a[0]);innovation=max(innovation,float(np.linalg.norm(b[1]-predicted)))
                corrupt=bool(cameras and enough and not protected and innovation>margin)
                if corrupt:retire.append(row['identity'])
                audits.append(dict(neural_id=row['identity'],index=row['index'],label=row['label'],visible_center=row['visible_center'],support_pixels=row['support_pixels'],occluded_pixels=row['occluded_pixels'],reliable_pixels=row['reliable_pixels'],protected_observation=protected,history_observations=len(h),history_seconds=now-h[0][0],motion_innovation_m=innovation,untrusted_geometry=corrupt,duplicate_group_size=len(indices)))
        self.records={k:v for k,v in self.records.items() if now-v['last_seen']<=cfg['max_gap_seconds']}
        return dict(policy='unobserved_inconsistent_proposal_evidence_v1',reset=reset,untrusted_geometry=sorted(set(retire)),queries=audits,current_detections_filtered=False,obstacle_memory_modified=False,free_space_asserted=False)
