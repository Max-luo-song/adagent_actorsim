"""Bounded road/observed-actor gate. No actor truth, route or scene-specific IDs."""
import copy,time
import numpy as np
from map_reference import MapRoadReference
from planning_guard import rectangle_separation,sweep_allowance

POLICY=dict(position_margin_m=.6,stale_growth_mps=.5,confirmation_hits=3,
            confirmation_span_s=.19,unlatched_expiry_s=1.,release_hits=3,
            maximum_reference_projection_iterations=6,nominal_reaction_steps=2)

class JointReference(MapRoadReference):
    def __init__(self,vehicle,mapping):
        super().__init__(vehicle,mapping)
        self.tracks=[];self.clock=None;self.scene=None;self.next_id=0

    def process(self,raw,arrays,obs,info,motion):
        began=time.perf_counter();raw=np.asarray(raw,float)
        if raw.shape!=(6,2) or not np.isfinite(raw).all():raise ValueError('Invalid current plan')
        self.observed(obs,info,arrays);adopted=raw.copy()
        before=self.certificate(self.states(raw))
        # Never penalize a legal lateral avoidance trajectory for leaving its
        # initial edge distance. Correct only boundary violations, then recheck.
        iterations=0
        while not self.certificate(self.states(adopted))['passed'] and iterations<POLICY['maximum_reference_projection_iterations']:
            states=self.states(adopted);pad=float(np.max(sweep_allowance(states,self.v)))
            for i in range(6):
                rows=states[max(0,i*5):min(len(states),(i+1)*5+2)]
                for edge in self.packet['edges']:
                    a,b=edge['a'],edge['b'];theta=rows[:,2]
                    support=np.max(self.v.half_width_m*np.abs(np.cos(theta)+a*np.sin(theta))+self.v.half_length_m*np.abs(np.sin(theta)-a*np.cos(theta)))
                    reserve=(self.v.clearance_margin_m+self.geometry.uncertainty+.06+pad)*np.hypot(1,a)
                    boundary=a*adopted[i,1]+b
                    if edge['side']=='right':adopted[i,0]=min(adopted[i,0],boundary-support-reserve)
                    else:adopted[i,0]=max(adopted[i,0],boundary+support+reserve)
            iterations+=1
        self.reference_states=self.states(adopted);cert=self.certificate(self.reference_states);self.reference_valid=cert['passed']
        self.update_tracks(arrays,info)
        return adopted,dict(source='joint_road_actor_gate_v7',reference_changed=bool(np.any(adopted!=raw)),
            legal_native_reference_preserved=bool(before['passed'] and np.array_equal(raw,adopted)),
            reference_certificate=cert,original_certificate=before,projection_iterations=iterations,
            initial_edge_alignment_used=False,seconds=time.perf_counter()-began,policy=POLICY)

    def update_tracks(self,arrays,info):
        now=float(info['timestamp']);scene=info.get('scene_token')
        if self.clock is not None and (now<self.clock or scene!=self.scene):self.tracks=[]
        if self.clock is not None and now==self.clock:raise ValueError('Repeated observation cannot renew actor evidence')
        self.clock=now;self.scene=scene
        boxes=np.asarray(arrays['boxes'],float);scores=np.asarray(arrays['scores'],float);labels=np.asarray(arrays['labels'])
        if boxes.ndim!=2 or boxes.shape[1]!=9 or scores.shape!=(len(boxes),) or labels.shape!=(len(boxes),) or not np.isfinite(boxes).all() or not np.isfinite(scores).all() or np.any(boxes[:,3:6]<=0):raise ValueError('Invalid actor evidence; absence is not clearance')
        used=set();accepted=[]
        for i in np.argsort(-scores):
            if scores[i]<.35:continue
            box=boxes[i].copy()
            if any(np.linalg.norm(box[:2]-b[:2])<1. for b in accepted):continue
            accepted.append(box)
            world=self.geometry.world(box[None,:2])[0]
            tip=self.geometry.world((box[:2]+np.array([np.sin(box[6]),np.cos(box[6])]))[None])[0]
            choices=[t for t in self.tracks if t['id'] not in used and t['label']==int(labels[i]) and now-t['last']<1. and np.linalg.norm(world-t['world'])<2.+12.*(now-t['last'])]
            track=min(choices,key=lambda t:np.linalg.norm(world-t['world'])) if choices else None
            if track is None:
                track=dict(id=self.next_id,label=int(labels[i]),samples=[],latched=False,clear_hits=0);self.next_id+=1;self.tracks.append(track)
            track.update(world=world,direction=(tip-world)/np.linalg.norm(tip-world),box=box,last=now)
            track['samples']=[(t,p) for t,p in track['samples'] if now-t<=.8]+[(now,world.copy())]
            used.add(track['id'])
        self.tracks=[t for t in self.tracks if t['latched'] or now-t['last']<=POLICY['unlatched_expiry_s']]

    def actor_boxes(self):
        boxes=[];active=[]
        for track in self.tracks:
            samples=track['samples'];span=samples[-1][0]-samples[0][0]
            if not track['latched'] and (len(samples)<POLICY['confirmation_hits'] or span<POLICY['confirmation_span_s']):continue
            age=self.clock-track['last'];velocity=np.zeros(2)
            if span>=.39:
                ts=np.array([t for t,_ in samples]);pts=np.array([p for _,p in samples]);ts-=ts.mean()
                velocity=(ts[:,None]*(pts-pts.mean(0))).sum(0)/(ts@ts)
                # Fit observed world displacement, never the single-frame
                # neural velocity which was spuriously 14m/s in the failure.
                velocity*=min(1.,30./max(np.linalg.norm(velocity),1e-9))
            world=track['world']+velocity*min(age,1.)
            xy=self.geometry.model(world[None])[0];tip=self.geometry.model((world+track['direction'])[None])[0]
            local_v=self.geometry.model((world+velocity)[None])[0]-xy
            box=track['box'].copy();box[:2]=xy;box[6]=np.arctan2(*(tip-xy));box[7:]=local_v
            pad=POLICY['position_margin_m']+POLICY['stale_growth_mps']*age
            box[3:5]+=2*pad;boxes.append(box);active.append(track)
        return np.asarray(boxes).reshape(-1,9),active

    def actor_certificate(self,states,boxes):
        if not len(boxes):return dict(passed=True,minimum_separation_m=None)
        times=np.arange(len(states))*self.v.dt
        separation=rectangle_separation(states,boxes,times,self.v.half_width_m,self.v.half_length_m)
        gap=float(np.min(separation-sweep_allowance(states,self.v)[:,None])-self.v.clearance_margin_m)
        return dict(passed=gap>=.05,minimum_separation_m=gap)

    def stop_states(self,motion,action,hold=False,reaction_steps=1):
        m=copy.copy(motion);origin=np.array([motion.z,motion.x]);yaw=motion.yaw;rows=[]
        def record():
            dz,dx=np.array([m.z,m.x])-origin;c,s=np.cos(yaw),np.sin(yaw)
            rows.append([c*dx-s*dz-self.mount[0,3],c*dz+s*dx-self.mount[1,3],m.yaw-yaw,m.speed,m.steer])
        record()
        for _ in range(reaction_steps):m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']));record()
        for _ in range(int(np.ceil(m.speed/(-self.v.min_acc_mps2*self.v.dt)))+2):
            if m.speed<=1e-9:break
            rate=0. if hold else float(np.clip(-m.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees)))
            m.advance(dict(acc=max(self.v.min_acc_mps2,-m.speed/self.v.dt),steer_rate=rate));record()
        return np.asarray(rows)

    def admit(self,motion,nominal):
        began=time.perf_counter();boxes,tracks=self.actor_boxes()
        def certificate(states):
            road=self.certificate(states);actors=self.actor_certificate(states,boxes)
            return dict(passed=road['passed'] and actors['passed'],road=road,actors=actors,physical_collision_free_certified=False)
        reference=self.actor_certificate(self.reference_states,boxes)
        nominal_cert=certificate(self.stop_states(motion,nominal,reaction_steps=POLICY['nominal_reaction_steps']))
        # A positive current conflict latches risk. Missing boxes/occupancy
        # never refresh a lease or count as a fresh clear observation.
        for box,track in zip(boxes,tracks):
            unsafe=not self.actor_certificate(self.reference_states,box[None])['passed']
            if unsafe:track['latched']=True;track['clear_hits']=0
            elif track['last']==self.clock and nominal_cert['passed']:
                track['clear_hits']+=1
                if track['clear_hits']>=POLICY['release_hits']:track['latched']=False
            else:track['clear_hits']=0
        latched=any(t['latched'] for t in self.tracks)
        audit=dict(nominal=nominal,nominal_certificate=nominal_cert,reference_actor_certificate=reference,
            risk_latched=latched,actor_evidence=[dict(id=t['id'],age_s=self.clock-t['last'],latched=t['latched'],clear_hits=t['clear_hits']) for t in tracks],
            heavy_mpc_executed=False,ground_truth_actor_used=False)
        if self.reference_valid and reference['passed'] and nominal_cert['passed'] and not latched:
            audit.update(selected='joint_verified_tracker',override=False,certificate=nominal_cert,seconds=time.perf_counter()-began);return nominal,audit
        candidates=[]
        for hold in (False,True):
            brake=dict(acc_mps2=max(self.v.min_acc_mps2,-motion.speed/self.v.dt),steer_rate_radps=0. if hold else float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
            cert=certificate(self.stop_states(motion,brake,hold=hold));candidates.append(cert)
            if cert['passed']:
                audit.update(selected='joint_verified_brake_hold' if hold else 'joint_verified_brake_unwind',override=True,certificate=cert,seconds=time.perf_counter()-began);return brake,audit
        audit.update(selected=None,override=True,reason='no_jointly_verified_action',brake_rejections=candidates,seconds=time.perf_counter()-began)
        return None,audit
