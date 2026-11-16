"""Bounded joint corridor/velocity repair; existing iLQR executes the chosen prefix."""
import copy,time
import numpy as np
from joint_safety_v7 import JointReference as MemoryGate
from planning_guard import rectangle_separation,sweep_allowance

POLICY=dict(max_preview_s=12.,min_preview_s=5.,forecast_support_s=2.5,
            stationary_compatible_speed_mps=1.5,stationary_position_span_m=1.2,
            future_lateral_growth_mps=.12,track_residual_margin_m=.2,
            tracking_reserve_m=.25,jerk_mps3=3.,max_candidates=15,
            missing_risk_acceleration_lease_s=.3,confirmation_hits=3,
            confirmation_span_s=.19,unlatched_expiry_s=1.)

def smooth(x):
    x=np.clip(x,0.,1.);return x*x*x*(10+x*(-15+6*x))

class JointReference(MemoryGate):
    def __init__(self,vehicle,mapping):
        super().__init__(vehicle,mapping);self.side=None;self.selected=None

    def update_tracks(self,arrays,info):
        reset=self.clock is not None and (float(info['timestamp'])<self.clock or info.get('scene_token')!=self.scene)
        super().update_tracks(arrays,info)
        if reset:self.side=None;self.selected=None

    def actor_boxes(self):
        boxes,tracks=super().actor_boxes();self.motion_evidence=[]
        for b,t in zip(boxes,tracks):
            samples=t['samples'];pts=np.array([p for _,p in samples]);span=samples[-1][0]-samples[0][0]
            stationary=span>=.39 and np.linalg.norm(b[7:])<POLICY['stationary_compatible_speed_mps'] and np.max(np.linalg.norm(pts-np.median(pts,axis=0),axis=1))<POLICY['stationary_position_span_m']
            if stationary:b[7:]=0.
            self.motion_evidence.append(dict(id=t['id'],stationary_compatible=bool(stationary),age_s=self.clock-t['last'],velocity_model_mps=b[7:].tolist()))
        return boxes,tracks

    def actor_certificate(self,states,boxes):
        if not len(boxes):return dict(passed=True,minimum_separation_m=None)
        times=np.arange(len(states))*self.v.dt
        growth=POLICY['future_lateral_growth_mps']*np.maximum(times-POLICY['forecast_support_s'],0.)
        # The central fit is not a certified future. Check three bounded
        # velocity hypotheses and grow geometry beyond observed forecast time.
        minimum=np.inf
        for fraction in (0.,-1.,1.):
            hyp=boxes.copy();moving=np.linalg.norm(hyp[:,7:9],axis=1)>.01
            hyp[moving,8]+=fraction*.5
            separation=rectangle_separation(states,hyp,times,self.v.half_width_m,self.v.half_length_m)
            temporal=.5*self.v.dt*np.linalg.norm(hyp[:,7:9],axis=1)
            gap=separation-sweep_allowance(states,self.v)[:,None]-growth[:,None]-temporal[None]-self.v.clearance_margin_m-POLICY['track_residual_margin_m']
            minimum=min(minimum,float(gap.min()))
        return dict(passed=minimum>=.05,minimum_separation_m=minimum,forecast_support_s=POLICY['forecast_support_s'],beyond_forecast_uncertainty_grows=True,physical_collision_free_certified=False)

    def joint_certificate(self,states,boxes):
        road=self.certificate(states);actors=self.actor_certificate(states,boxes)
        s=np.asarray(states);dv=np.diff(s[:,3])/self.v.dt;dr=np.diff(s[:,4])/self.v.dt
        dynamics=bool(np.isfinite(s).all() and np.all(s[:,3]>=-1e-8) and np.all(s[:,3]<=self.v.max_speed_mps+1e-8) and np.all(np.abs(s[:,4])<=np.radians(self.v.max_steer_degrees)+1e-8) and np.all(dv>=self.v.min_acc_mps2-1e-7) and np.all(dv<=self.v.max_acc_mps2+1e-7) and np.all(np.abs(dr)<=np.radians(self.v.max_steer_rate_degrees)+1e-7))
        return dict(passed=road['passed'] and actors['passed'] and dynamics,road=road,actors=actors,dynamics_bounds_passed=dynamics)

    def bounds(self,y,speed=0.):
        lo=np.full_like(np.asarray(y,float),-np.inf);hi=np.full_like(lo,np.inf)
        reserve=self.v.half_width_m+self.v.clearance_margin_m+self.geometry.uncertainty+POLICY['tracking_reserve_m']+.05+.5*self.v.dt*speed
        for e in self.packet['edges']:
            edge=e['a']*y+e['b'];pad=reserve*np.hypot(1,e['a'])
            if e['side']=='right':hi=np.minimum(hi,edge-pad)
            else:lo=np.maximum(lo,edge+pad)
        return lo,hi

    def guides(self,raw,boxes,motion,target):
        origin=-self.mount[:2,3];length=max(30.,target*POLICY['max_preview_s']+10.)
        y=np.arange(origin[1],origin[1]+length+.2,.2)
        knots=np.r_[origin[1],raw[:,1]];xs=np.r_[origin[0],raw[:,0]]
        # Forward guides need monotonic stations. Non-forward raw plans are
        # retained only through the original-plan gate, never reordered here.
        if np.any(np.diff(knots)<=0):base=np.full_like(y,origin[0])
        else:
            base=np.interp(y,knots,xs)
            tail=y>knots[-1]
            terminal_slope=(xs[-1]-xs[-2])/(knots[-1]-knots[-2])
            base[tail]=xs[-1]+terminal_slope*(y[tail]-knots[-1])
        lo,hi=self.bounds(y,max(target,motion.speed));base=np.clip(base,lo,hi)
        join_length=max(10.,target*3.)
        u=np.clip((y-origin[1])/join_length,0.,1.)
        initial_curvature=np.tan(motion.steer)/self.v.wheelbase_m
        base=origin[0]+(base-origin[0])*smooth(u)+.5*initial_curvature*join_length**2*u*u*(1-u)**3
        base=np.clip(base,lo,hi);base[0]=origin[0]
        result=[dict(side=0,y=y,x=base,end_y=origin[1]+max(15.,target*5.),clear_y=None)]
        blockers=[]
        for b in boxes:
            if b[1]+b[4]/2+self.v.half_length_m<origin[1]:continue
            if b[1]>y[-1]:continue
            stations=np.linspace(max(origin[1],b[1]-b[4]/2-self.v.half_length_m-1.),max(origin[1],b[1]+b[4]/2+self.v.half_length_m+1.),7)
            lane=np.interp(stations,y,base)
            offset=float(np.min(np.abs(b[0]-lane)))
            if offset<b[3]/2+self.v.half_width_m+self.v.clearance_margin_m+.8:blockers.append(b)
        if not blockers:
            # A road-parallel recovery seed can unwind heading before the
            # current neural reference can be joined. It holds CURRENT
            # lateral position and is never an initial-frame lane anchor.
            road_slope=float(np.mean([e['a'] for e in self.packet['edges']]))
            parallel=origin[0]+road_slope*(y-origin[1])
            result.append(dict(side=0,y=y,x=parallel,end_y=result[0]['end_y'],clear_y=None,lookahead_scale=1.8))
            return result
        b=min(blockers,key=lambda b:b[1]);result[0]['clear_y']=float(b[1]+b[4]/2+self.v.half_length_m+1.)
        near=b[1]-b[4]/2-self.v.half_length_m-.5
        shift_end=max(origin[1]+3.,near-1.)
        return_start=b[1]+b[4]/2+self.v.half_length_m+1.
        return_length=max(8.,target*2.5)
        for side in (-1,1):
            future_pad=POLICY['future_lateral_growth_mps']*max((return_start-origin[1])/max(target,1.)-POLICY['forecast_support_s'],0.)
            goal=b[0]+side*(b[3]/2+self.v.half_width_m+self.v.clearance_margin_m+POLICY['tracking_reserve_m']+.55+future_pad)
            u=np.clip((y-origin[1])/max(shift_end-origin[1],1.),0.,1.)
            initial_curvature=np.tan(motion.steer)/self.v.wheelbase_m
            outbound=origin[0]+(goal-origin[0])*smooth(u)+.5*initial_curvature*(shift_end-origin[1])**2*u*u*(1-u)**3
            returning=smooth((y-return_start)/return_length)
            x=outbound*(1-returning)+base*returning
            # A larger future tracking reserve need not hold at t=0.
            # Reject on the exact executed-footprint certificate below,
            # not on this guide-only prefilter.
            for lookahead_scale in (1.2,1.8):
                result.append(dict(side=side,y=y,x=x,end_y=return_start+return_length,clear_y=return_start,lookahead_scale=lookahead_scale))
        return result

    def rollout(self,guide,motion,target,horizon):
        dt=self.v.dt;n=int(round(horizon/dt));s=np.array([-self.mount[0,3],-self.mount[1,3],0.,motion.speed,motion.steer],float)
        slope=np.gradient(guide['x'],guide['y']);curvature=np.gradient(slope,guide['y'])/(1+slope*slope)**1.5
        rows=[s.copy()];controls=[];last=float(getattr(motion,'last_accel',0.))
        rate_limit=np.radians(self.v.max_steer_rate_degrees);steer_limit=np.radians(self.v.max_steer_degrees)
        for _ in range(n):
            ahead=max(3.,guide.get('lookahead_scale',1.2)*max(s[3],1.))
            gy=s[1]+ahead;gx=float(np.interp(gy,guide['y'],guide['x']));dx,dy=gx-s[0],gy-s[1]
            right=np.cos(s[2])*dx-np.sin(s[2])*dy
            desired=np.arctan2(2*self.v.wheelbase_m*right,max(dx*dx+dy*dy,1e-6))
            rate=float(np.clip((desired-s[4])/dt,-rate_limit,rate_limit))
            # Curvature determines local speed; steering rate is enforced
            # above. Look ahead before entering a tight bend.
            near=(guide['y']>=s[1])&(guide['y']<=s[1]+max(3.,s[3]*1.5))
            max_curv=float(np.max(np.abs(curvature[near]))) if near.any() else 0.
            safe_speed=min(target,np.sqrt(2./max(max_curv,1e-6)))
            acc=float(np.clip(1.5*(safe_speed-s[3]),max(self.v.min_acc_mps2,last-POLICY['jerk_mps3']*dt),min(self.v.max_acc_mps2,last+POLICY['jerk_mps3']*dt)))
            acc=max(acc,-s[3]/dt);s[3]=np.clip(s[3]+acc*dt,0,self.v.max_speed_mps)
            s[4]=np.clip(s[4]+rate*dt,-steer_limit,steer_limit)
            s[0]+=s[3]*np.sin(s[2])*dt;s[1]+=s[3]*np.cos(s[2])*dt
            s[2]+=s[3]*np.tan(s[4])/self.v.wheelbase_m*dt
            rows.append(s.copy());controls.append([acc,rate]);last=acc
        return np.array(rows),np.array(controls)

    def terminal_stop(self,states):
        s=states[-1].copy();rows=[];rate_limit=np.radians(self.v.max_steer_rate_degrees)
        for _ in range(int(np.ceil(s[3]/(-self.v.min_acc_mps2*self.v.dt)))+2):
            if s[3]<=1e-8:break
            s[3]=max(0.,s[3]+self.v.min_acc_mps2*self.v.dt)
            s[4]+=np.clip(-s[4],-rate_limit*self.v.dt,rate_limit*self.v.dt)
            s[0]+=s[3]*np.sin(s[2])*self.v.dt;s[1]+=s[3]*np.cos(s[2])*self.v.dt
            s[2]+=s[3]*np.tan(s[4])/self.v.wheelbase_m*self.v.dt;rows.append(s.copy())
        return np.vstack([states,rows]) if len(rows) else states

    def process(self,raw,arrays,obs,info,motion):
        began=time.perf_counter();raw=np.asarray(raw,float)
        if raw.shape!=(6,2) or not np.isfinite(raw).all():raise ValueError('Invalid current plan')
        self.observed(obs,info,arrays);self.update_tracks(arrays,info);boxes,tracks=self.actor_boxes()
        original_states=self.states(raw);before=self.joint_certificate(original_states,boxes)
        target=float(np.clip(np.median(original_states[1:,3]),0,self.v.max_speed_mps))
        if not np.isfinite(target):raise ValueError('Invalid target speed')
        reachable=abs(original_states[0,3]-motion.speed)<=.5 and np.max(np.abs(np.diff(original_states[:,3]))/self.v.dt)<=max(-self.v.min_acc_mps2,self.v.max_acc_mps2)+.1
        # Risk is retained independently of the selected route's short horizon.
        for b,t in zip(boxes,tracks):
            relevant=b[1]+b[4]/2>-self.mount[1,3] and abs(b[0])<b[3]/2+self.v.half_width_m+.8
            if relevant:t['latched']=True
            if b[1]+b[4]/2+self.v.half_length_m+.5<-self.mount[1,3] :t['latched']=False
        active=any(t['latched'] for t in self.tracks)
        stale=any(t['latched'] and self.clock-t['last']>POLICY['missing_risk_acceleration_lease_s'] for t in self.tracks)
        candidates=[];self.selected=None
        # Preserve original reference only with no unresolved front conflict.
        if before['passed'] and reachable and not active:
            self.selected_states=None;self.reference_states=original_states;self.reference_valid=True;adopted=raw.copy();mode='native_preserved'
        else:
            for fraction in (1.,.75,.5):
                speed=target*fraction
                for guide in self.guides(raw,boxes,motion,speed):
                    horizon=POLICY['max_preview_s'] if guide['clear_y'] is not None else float(np.clip((guide['end_y']+self.mount[1,3])/max(speed,motion.speed,1.),POLICY['min_preview_s'],POLICY['max_preview_s']))
                    states,controls=self.rollout(guide,motion,speed,horizon)
                    full=self.terminal_stop(states);cert=self.joint_certificate(full,boxes)
                    # iLQR consumes the same dense physical prefix. Six positions
                    # are exported solely for existing native-score interfaces.
                    nodes=states[np.arange(5,31,5),:2]
                    pc=self.joint_certificate(states[:31],boxes)
                    score=float(2*np.mean((states[:31,:2]-original_states[:31,:2])**2)+.2*np.mean(controls[:,0]**2)+.1*np.mean(controls[:,1]**2)-6*(states[-1,1]-states[0,1])/horizon)
                    if self.side and guide['side'] and guide['side']!=self.side:score+=5.
                    complete=guide['clear_y'] is None or states[-1,1]>=guide['clear_y']
                    item=dict(clear_conflict_in_preview=bool(complete),side=guide['side'],speed_mps=speed,horizon_s=horizon,certificate=cert,prefix_certificate=pc,score=score,passed=bool(cert['passed'] and pc['passed']),progress_m=float(states[-1,1]-states[0,1]))
                    if item['passed'] and not stale:candidates.append((score,nodes,states,controls,item))
                    self._audit_rows=getattr(self,'_audit_rows',[])+[item]
            if candidates:
                _,adopted,states,controls,chosen=min(candidates,key=lambda x:x[0]);self.selected_controls=controls;self.selected=chosen;self.selected_states=states;self.side=chosen['side'] or self.side
                self.reference_states=states[:31].copy();self.reference_valid=True;mode='joint_corridor_velocity_repair'
            else:
                self.selected_states=None
                brake=dict(acc_mps2=max(self.v.min_acc_mps2,-motion.speed/self.v.dt),steer_rate_radps=float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
                stop=self.stop_states(motion,brake);padded=np.vstack([stop,np.repeat(stop[-1:],max(0,31-len(stop)),axis=0)])
                adopted=padded[np.arange(5,31,5),:2];self.reference_states=self.states(adopted)
                self.reference_valid=self.joint_certificate(stop,boxes)['passed'];mode='verified_wait_no_feasible_corridor'
        audit=dict(source='joint_road_actor_repair_v8',mode=mode,reference_changed=not np.array_equal(raw,adopted),legal_native_reference_preserved=mode=='native_preserved',reference_certificate=self.certificate(self.reference_states),original_certificate=before['road'],selected_candidate=self.selected,tracking_reference_source='dense_kinematic_states' if self.selected is not None else 'native_or_braking',selected_rollout_states=self.selected_states.tolist() if self.selected_states is not None else None,candidates=getattr(self,'_audit_rows',[]),risk_retained=active,stale_risk=stale,motion_evidence=self.motion_evidence,policy=POLICY,seconds=time.perf_counter()-began,initial_edge_alignment_used=False)
        self._audit_rows=[];self.last_selection=audit
        return adopted,audit

    def continuation(self,motion,nominal):
        s=np.array([-self.mount[0,3],-self.mount[1,3],0.,motion.speed,motion.steer]);rows=[s.copy()]
        controls=self.selected_controls.copy();controls[0]=[nominal['acc_mps2'],nominal['steer_rate_radps']]
        for acc,rate in controls:
            s[3]=max(0.,s[3]+acc*self.v.dt);s[4]+=rate*self.v.dt
            s[0]+=s[3]*np.sin(s[2])*self.v.dt;s[1]+=s[3]*np.cos(s[2])*self.v.dt
            s[2]+=s[3]*np.tan(s[4])/self.v.wheelbase_m*self.v.dt;rows.append(s.copy())
        return self.terminal_stop(np.array(rows))

    def admit(self,motion,nominal):
        began=time.perf_counter();boxes,tracks=self.actor_boxes()
        original_nominal=nominal.copy()
        if self.selected_states is not None:
            ceiling=(self.selected_states[1,3]-motion.speed)/self.v.dt
            nominal=dict(nominal,acc_mps2=min(nominal['acc_mps2'],float(ceiling)))
        stop=self.stop_states(motion,nominal,reaction_steps=2);cert=self.joint_certificate(stop,boxes)
        reference=self.actor_certificate(self.reference_states,boxes)
        stale=any(t['latched'] and self.clock-t['last']>POLICY['missing_risk_acceleration_lease_s'] for t in self.tracks)
        continuation=None
        if not cert['passed'] and self.selected is not None:
            continuation=self.joint_certificate(self.continuation(motion,nominal),boxes)
        witness=cert if cert['passed'] else continuation
        allow=self.reference_valid and reference['passed'] and witness is not None and witness['passed'] and self.last_selection['mode']!='verified_wait_no_feasible_corridor' and not(stale and nominal['acc_mps2']>0)
        audit=dict(nominal=original_nominal,velocity_envelope_action=nominal,nominal_certificate=cert,continuation_certificate=continuation,reference_actor_certificate=reference,risk_latched=any(t['latched'] for t in self.tracks),actor_evidence=[dict(id=t['id'],age_s=self.clock-t['last'],latched=t['latched']) for t in tracks],ground_truth_actor_used=False,heavy_mpc_executed=False,selected_candidate=self.selected)
        if allow:
            audit.update(selected='joint_verified_tracker',override=nominal!=original_nominal,certificate=witness,safe_witness='immediate_stop' if cert['passed'] else 'tracking_continuation_with_terminal_stop',seconds=time.perf_counter()-began);return nominal,audit
        for hold in (False,True):
            brake=dict(acc_mps2=max(self.v.min_acc_mps2,-motion.speed/self.v.dt),steer_rate_radps=0. if hold else float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
            cert=self.joint_certificate(self.stop_states(motion,brake,hold=hold),boxes)
            if cert['passed']:
                audit.update(selected='joint_verified_brake_hold' if hold else 'joint_verified_brake_unwind',override=True,certificate=cert,seconds=time.perf_counter()-began);return brake,audit
        audit.update(selected=None,override=True,reason='no_jointly_verified_action',seconds=time.perf_counter()-began);return None,audit
