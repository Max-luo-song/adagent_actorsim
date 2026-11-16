"""Observed-track lifecycle. Identity expiry never asserts observed free space."""
import numpy as np
from shapely.geometry import Polygon
from observation_evidence import update_motion,EVIDENCE_POLICY

MEMORY_POLICY=dict(association_base_m=2.,association_growth_mps=1.,association_growth_limit_m=6.,
                   duplicate_iou=.6,dimension_ratio_limit=1.8,heading_cosine_min=.5,
                   confirmation_hits=3,confirmation_span_s=.19,sample_window_s=1.6,
                   inactive_expiry_s=1.,position_margin_m=.6,stale_growth_mps=.5,
                   maximum_fit_speed_mps=30.,stationary_speed_mps=1.5)

def footprint(center,direction,box):
    forward=np.asarray(direction);right=np.array([forward[1],-forward[0]])
    return Polygon(np.asarray(center)+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])@np.stack([right*box[3]/2,forward*box[4]/2]))

def compatible(track,det):
    ratio=np.maximum(track['box'][3:5]/det['box'][3:5],det['box'][3:5]/track['box'][3:5])
    return track['label']==det['label'] and np.max(ratio)<=MEMORY_POLICY['dimension_ratio_limit'] and abs(track['direction']@det['direction'])>=MEMORY_POLICY['heading_cosine_min']

def predict(track,now):
    return track['world']+track.get('velocity',np.zeros(2))*max(0.,now-track['last'])

def fit(track):
    update_motion(track)

class RiskMemory:
    def update_tracks(self,arrays,info):
        now=float(info['timestamp']);scene=info.get('scene_token')
        if not np.isfinite(now):raise ValueError('Nonfinite observation time')
        reset=self.clock is not None and (now<self.clock or scene!=self.scene)
        if reset:self.tracks=[];self.side=None;self.selected=None;self.previous_target_speed=None
        if self.clock is not None and not reset and now==self.clock:raise ValueError('Repeated observation cannot renew evidence')
        self.clock=now;self.scene=scene
        boxes=np.asarray(arrays['boxes'],float);scores=np.asarray(arrays['scores'],float);labels=np.asarray(arrays['labels'])
        if boxes.ndim!=2 or boxes.shape[1]!=9 or scores.shape!=(len(boxes),) or labels.shape!=(len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any(boxes[:,3:6]<=0):raise ValueError('Invalid actor evidence')
        self.memory_audit=dict(reset=reset,reassociated=[],merged=[],retired=[],new=[])
        detections=[]
        for i in np.argsort(-scores):
            if scores[i]<.35:continue
            b=boxes[i].copy();world=self.geometry.world(b[None,:2])[0]
            tip=self.geometry.world((b[:2]+np.array([np.sin(b[6]),np.cos(b[6])]))[None])[0];direction=(tip-world)/np.linalg.norm(tip-world)
            det=dict(box=b,world=world,direction=direction,label=int(labels[i]));poly=footprint(world,direction,b)
            # Preserve the reviewed current-frame deduplication and tracker
            # geometry. This patch changes stale lifecycle, not fresh boxes.
            if not any(np.linalg.norm(det['world']-prev['world'])<1. for prev in detections):detections.append(det)
        used=set();merged=set()
        for det in detections:
            choices=[]
            for t in self.tracks:
                if t['id'] in used or t['id'] in merged or t['label']!=det['label']:continue
                age=now-t['last']
                if age<1.:
                    distance=float(np.linalg.norm(t['world']-det['world']))
                    if distance<2.+12.*age:choices.append((distance,t))
                elif t['latched'] and compatible(t,det):
                    distance=float(np.linalg.norm(predict(t,now)-det['world']))
                    gate=MEMORY_POLICY['association_base_m']+min(MEMORY_POLICY['association_growth_limit_m'],MEMORY_POLICY['association_growth_mps']*age)
                    if distance<=gate:choices.append((distance,t))
            track=min(choices,key=lambda x:x[0])[1] if choices else None
            if track is None:
                track=dict(id=self.next_id,label=det['label'],samples=[],latched=False,confirmed=False,first=now,last=now,velocity=np.zeros(2),position_growth_mps=.5)
                self.next_id+=1;self.tracks.append(track);self.memory_audit['new'].append(track['id'])
            gap=now-track['last']
            if gap>1.:
                self.memory_audit['reassociated'].append(dict(id=track['id'],gap_s=gap))
                # A long-gap reappearance does not certify the old velocity.
                track['velocity']=np.zeros(2)
            # Merge only shape-compatible overlapping identities, never just
            # nearby vehicles. Keep one identity and one observation timestamp.
            current_poly=footprint(det['world'],det['direction'],det['box'])
            for other in self.tracks:
                if now-other['last']<=1. or other is track or other['id'] in used or other['id'] in merged or not other.get('confirmed') or not compatible(other,det):continue
                old_poly=footprint(predict(other,now),other['direction'],other['box']);union=current_poly.union(old_poly).area
                if union and current_poly.intersection(old_poly).area/union>=MEMORY_POLICY['duplicate_iou']:
                    track['latched']|=other['latched'];track['confirmed']|=other['confirmed'];merged.add(other['id'])
                    self.memory_audit['merged'].append(dict(old=other['id'],into=track['id']))
            track.update(world=det['world'],direction=det['direction'],box=det['box'],last=now,declared_ground_z=-float(np.asarray(info['model_lidar_frame']['virtual_to_ego'])[2,3]))
            track['samples']=[(t,p) for t,p in track['samples'] if now-t<=MEMORY_POLICY['sample_window_s']]+[(now,det['world'].copy())]
            if len(track['samples'])>=MEMORY_POLICY['confirmation_hits'] and track['samples'][-1][0]-track['samples'][0][0]>=MEMORY_POLICY['confirmation_span_s']:track['confirmed']=True
            fit(track);used.add(track['id'])
        retained=[]
        for t in self.tracks:
            if t['id'] in merged:continue
            if not t['latched'] and now-t['last']>MEMORY_POLICY['inactive_expiry_s']:
                self.memory_audit['retired'].append(dict(id=t['id'],reason='inactive_unlatched_identity_expired',free_space_claim=False));continue
            retained.append(t)
        self.tracks=retained

    def actor_boxes(self):
        boxes=[];tracks=[];self.motion_evidence=[]
        for t in self.tracks:
            if not t['confirmed']:continue
            age=self.clock-t['last'];anchor=t.get('anchor_world',t['world']);world=(anchor+t['world'])/2+t['velocity']*age
            xy=self.geometry.model(world[None])[0];tip=self.geometry.model((world+t['direction'])[None])[0]
            velocity=self.geometry.model((world+t['velocity'])[None])[0]-xy
            b=t['box'].copy();b[:2]=xy;b[6]=np.arctan2(*(tip-xy));b[7:]=velocity
            pad=MEMORY_POLICY['position_margin_m']
            right=np.array([t['direction'][1],-t['direction'][0]]);delta=anchor-t['world']
            b[3:5]+=np.abs([delta@right,delta@t['direction']])+2*pad
            boxes.append(b);tracks.append(t)
            self.motion_evidence.append(dict(id=t['id'],stationary_compatible=t.get('stationary_compatible',False),age_s=age,velocity_model_mps=velocity.tolist(),evidence_timestamp_s=t['last'],position_margin_m=pad,evidence_tier=t['evidence_tier'],height_supported=t['height_supported'],motion=t['motion_audit'],age_does_not_expand_observed_core=True,soft_age_margin_m=min(EVIDENCE_POLICY['soft_age_margin_limit_m'],EVIDENCE_POLICY['soft_age_growth_mps']*age)))
        return np.asarray(boxes).reshape(-1,9),tracks
