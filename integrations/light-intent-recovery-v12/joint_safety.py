"""Bounded joint corridor/velocity repair; existing iLQR executes the chosen prefix."""
import copy,time
import numpy as np
from joint_safety_v7 import JointReference as MemoryGate
from risk_memory import RiskMemory,MEMORY_POLICY
from observation_evidence import EVIDENCE_POLICY
from planning_guard import rectangle_separation,sweep_allowance

POLICY=dict(max_preview_s=12.,min_preview_s=5.,forecast_support_s=2.5,
            stationary_compatible_speed_mps=1.5,stationary_position_span_m=1.2,
            future_lateral_growth_mps=.12,track_residual_margin_m=.2,
            tracking_reserve_m=.25,jerk_mps3=3.,max_candidates=15,
            missing_risk_acceleration_lease_s=.3,confirmation_hits=3,
            confirmation_span_s=.19,unlatched_expiry_s=1.,target_continuity_weight=6.,side_switch_cost=2.,soft_risk_weight=2.,intent_recovery_hold_s=.5)

def smooth(x):
    x=np.clip(x,0.,1.);return x*x*x*(10+x*(-15+6*x))

class JointReference(MemoryGate):
    def __init__(self,vehicle,mapping):
        super().__init__(vehicle,mapping);self.side=None;self.selected=None;self.previous_target_speed=None;self.intent_clear_since=None;self.intent_last_clock=None;self.intent_recovery_audit={}

    def update_tracks(self,arrays,info):
        return RiskMemory.update_tracks(self,arrays,info)

    def actor_boxes(self):
        return RiskMemory.actor_boxes(self)

    def actor_certificate(self,states,boxes):
        if not len(boxes):return dict(passed=True,minimum_separation_m=None)
        times=np.arange(len(states))*self.v.dt
        growth=POLICY['future_lateral_growth_mps']*np.maximum(times-POLICY['forecast_support_s'],0.)
        # The central fit is not a certified future. Check three bounded
        # velocity hypotheses and grow geometry beyond observed forecast time.
        minimum=np.inf;per_actor=np.full(len(boxes),np.inf)
        for fraction in (0.,-1.,1.):
            hyp=boxes.copy();moving=np.linalg.norm(hyp[:,7:9],axis=1)>.01
            hyp[moving,8]+=fraction*.5
            separation=rectangle_separation(states,hyp,times,self.v.half_width_m,self.v.half_length_m)
            temporal=.5*self.v.dt*np.linalg.norm(hyp[:,7:9],axis=1)
            gap=separation-sweep_allowance(states,self.v)[:,None]-growth[:,None]-temporal[None]-self.v.clearance_margin_m-POLICY['track_residual_margin_m']
            minimum=min(minimum,float(gap.min()));per_actor=np.minimum(per_actor,gap.min(axis=0))
        return dict(passed=minimum>=.05,minimum_separation_m=minimum,per_actor_minimum_separation_m=per_actor.tolist(),forecast_support_s=POLICY['forecast_support_s'],beyond_forecast_uncertainty_grows=True,physical_collision_free_certified=False)

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
            # Heading recovery must not become a permanent lateral hold
            # once aligned: rejoin the CURRENT neural reference instead.
            if abs(np.arctan(road_slope))>.1 or abs(motion.steer)>.1:
                result.append(dict(side=0,y=y,x=parallel,end_y=result[0]['end_y'],clear_y=None,lookahead_scale=1.8))
            return result
        blockers=sorted(blockers,key=lambda b:b[1]);b=blockers[0]
        return_length=max(8.,target*2.5)
        cluster=[b];far=b[1]+b[4]/2+self.v.half_length_m+1.
        for other in blockers[1:]:
            near_other=other[1]-other[4]/2-self.v.half_length_m-.5
            if near_other<=far+return_length:
                cluster.append(other);far=max(far,other[1]+other[4]/2+self.v.half_length_m+1.)
        result[0]['clear_y']=float(far)
        near=min(x[1]-x[4]/2-self.v.half_length_m-.5 for x in cluster)
        shift_end=max(origin[1]+3.,near-1.);return_start=far
        road_slope=float(np.mean([e['a'] for e in self.packet['edges']]))
        for side in (-1,1):
            future_pad=POLICY['future_lateral_growth_mps']*max((return_start-origin[1])/max(target,1.)-POLICY['forecast_support_s'],0.)
            intercepts=[x[0]-road_slope*x[1]+side*(x[3]/2+self.v.half_width_m+self.v.clearance_margin_m+POLICY['tracking_reserve_m']+.55+future_pad)*np.hypot(1.,road_slope) for x in cluster]
            lane_intercept=min(intercepts) if side<0 else max(intercepts)
            lane_target=lane_intercept+road_slope*y
            u=np.clip((y-origin[1])/max(shift_end-origin[1],1.),0.,1.)
            initial_curvature=np.tan(motion.steer)/self.v.wheelbase_m
            outbound=origin[0]+(lane_target-origin[0])*smooth(u)+.5*initial_curvature*(shift_end-origin[1])**2*u*u*(1-u)**3
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
            acc=float(np.clip(acc,max(self.v.min_acc_mps2,-s[3]/dt),min(self.v.max_acc_mps2,(self.v.max_speed_mps-s[3])/dt)));s[3]+=acc*dt
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

    def speed_targets(self,cruise):
        previous=self.previous_target_speed
        warm=cruise*.75 if previous is None else previous+float(np.clip(cruise-previous,-self.v.max_acc_mps2*self.v.dt,self.v.max_acc_mps2*self.v.dt))
        if abs(warm-cruise)<.05:warm=cruise*.75
        return [float(cruise),float(np.clip(warm,0,cruise)),float(cruise*.5)]

    def uncertainty_cost(self,states,boxes,tracks):
        if not len(boxes):return 0.
        expanded=boxes.copy();confidence=[]
        for j,t in enumerate(tracks):
            age=max(0.,self.clock-t['last']);extra=min(EVIDENCE_POLICY['soft_age_margin_limit_m'],EVIDENCE_POLICY['soft_age_growth_mps']*age)
            expanded[j,3:5]+=2*extra
            confidence.append(np.exp(-age/EVIDENCE_POLICY['soft_confidence_decay_s'])*(1. if t.get('height_supported',True) else .25))
        gaps=rectangle_separation(states,expanded,np.arange(len(states))*self.v.dt,self.v.half_width_m,self.v.half_length_m)
        return float(POLICY['soft_risk_weight']*np.sum(np.mean(np.minimum(np.maximum(-gaps,0.)**2,4.),axis=0)*np.asarray(confidence)))

    def intent_pool(self,candidates):
        # Called once per actual observation, on ALREADY certified candidates.
        if self.intent_last_clock is not None and self.clock<=self.intent_last_clock:
            self.intent_clear_since=None
        self.intent_last_clock=self.clock
        current=[c for c in candidates if c[4]['side']==0 and c[4]['passed']]
        eligible=self.side in (-1,1) and bool(current)
        if not eligible:self.intent_clear_since=None
        elif self.intent_clear_since is None:self.intent_clear_since=self.clock
        stable=eligible and self.clock-self.intent_clear_since>=POLICY['intent_recovery_hold_s']-1e-9
        self.intent_recovery_audit=dict(eligible=bool(eligible),clear_since_s=self.intent_clear_since,
            resumed=bool(stable),hold_s=POLICY['intent_recovery_hold_s'],
            fully_feasible_candidates=len(candidates),current_intent_candidates=len(current),
            source='current neural intent; no initial-frame lane anchor')
        return current if stable else candidates

    def process(self,raw,arrays,obs,info,motion):
        began=time.perf_counter();raw=np.asarray(raw,float)
        if raw.shape!=(6,2) or not np.isfinite(raw).all():raise ValueError('Invalid current plan')
        self.observed(obs,info,arrays);self.update_tracks(arrays,info);boxes,tracks=self.actor_boxes()
        original_states=self.states(raw);before=self.joint_certificate(original_states,boxes)
        target=float(np.clip(np.median(original_states[1:,3]),0,self.v.max_speed_mps))
        if not np.isfinite(target):raise ValueError('Invalid target speed')
        reachable=abs(original_states[0,3]-motion.speed)<=.5 and np.max(np.abs(np.diff(original_states[:,3]))/self.v.dt)<=max(-self.v.min_acc_mps2,self.v.max_acc_mps2)+.1
        # Preview relevance triggers planning, not a persistent risk latch.
        # Only a CURRENT observation plus an actual geometric conflict can
        # create a latch. Age-driven inflation cannot create new evidence.
        self._risk_boxes=boxes;self._risk_tracks=tracks
        active=any(t['latched'] for t in self.tracks)
        for b,t in zip(boxes,tracks):
            frontal=b[1]+b[4]/2>-self.mount[1,3] and b[1]<max(motion.speed,target)*POLICY['max_preview_s']+10 and abs(b[0])<b[3]/2+self.v.half_width_m+.8
            if frontal:active=True
            behind=b[1]+b[4]/2+self.v.half_length_m+.5+max(0.,b[8]+.5)*POLICY['max_preview_s']<-self.mount[1,3]
            if behind:t['latched']=False
        for gap,t in zip(before['actors'].get('per_actor_minimum_separation_m',[]),tracks):
            if t['last']==self.clock and gap>=1.:t['latched']=False
        self.observe_conflicts(before['actors'])
        stale=any(t['latched'] and self.clock-t['last']>POLICY['missing_risk_acceleration_lease_s'] for t in self.tracks)
        candidates=[];self.selected=None
        # Preserve original reference only with no unresolved front conflict.
        if before['passed'] and reachable and not active:
            self.selected_states=None;self.reference_states=original_states;self.reference_valid=True;adopted=raw.copy();mode='native_preserved'
            self.intent_clear_since=None;self.intent_last_clock=self.clock;self.intent_recovery_audit=dict(resumed=False,reason='native_reference_preserved')
        else:
            for speed in self.speed_targets(target):
                for guide in self.guides(raw,boxes,motion,speed):
                    horizon=POLICY['max_preview_s'] if guide['clear_y'] is not None else float(np.clip((guide['end_y']+self.mount[1,3])/max(speed,motion.speed,1.),POLICY['min_preview_s'],POLICY['max_preview_s']))
                    states,controls=self.rollout(guide,motion,speed,horizon)
                    full=self.terminal_stop(states);cert=self.joint_certificate(full,boxes)
                    # Exploratory bypasses are proposals, not observed risk.
                    # Only current neural intent may establish a persistent
                    # obstacle obligation; rejected side paths cannot do so.
                    if guide['side']==0:self.observe_conflicts(cert['actors'])
                    # iLQR consumes the same dense physical prefix. Six positions
                    # are exported solely for existing native-score interfaces.
                    nodes=states[np.arange(5,31,5),:2]
                    pc=self.joint_certificate(states[:31],boxes)
                    score=float(2*np.mean((states[:31,:2]-original_states[:31,:2])**2)+.2*np.mean(controls[:,0]**2)+.1*np.mean(controls[:,1]**2)-6*(states[-1,1]-states[0,1])/horizon)
                    if self.side is not None and guide['side']!=self.side and any(t['latched'] for t in tracks):score+=POLICY['side_switch_cost']
                    continuity=0. if self.previous_target_speed is None else POLICY['target_continuity_weight']*(speed-self.previous_target_speed)**2
                    uncertain=self.uncertainty_cost(states,boxes,tracks);score+=continuity+uncertain
                    complete=guide['clear_y'] is None or states[-1,1]>=guide['clear_y']
                    item=dict(velocity_continuity_cost=continuity,uncertain_extent_cost=uncertain,clear_conflict_in_preview=bool(complete),side=guide['side'],speed_mps=speed,horizon_s=horizon,certificate=cert,prefix_certificate=pc,score=score,passed=bool(cert['passed'] and pc['passed']),progress_m=float(states[-1,1]-states[0,1]))
                    if item['passed']:candidates.append((score,nodes,states,controls,item))
                    self._audit_rows=getattr(self,'_audit_rows',[])+[item]
            candidates=self.intent_pool(candidates)
            if candidates:
                _,adopted,states,controls,chosen=min(candidates,key=lambda x:x[0]);self.selected_controls=controls;self.selected=chosen;self.selected_states=states;self.side=chosen['side'];self.previous_target_speed=chosen['speed_mps']
                self.reference_states=states[:31].copy();self.reference_valid=True;mode='joint_corridor_velocity_repair'
            else:
                self.selected_states=None
                brake=dict(acc_mps2=max(self.v.min_acc_mps2,-motion.speed/self.v.dt),steer_rate_radps=float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
                stop=self.stop_states(motion,brake);padded=np.vstack([stop,np.repeat(stop[-1:],max(0,31-len(stop)),axis=0)])
                adopted=padded[np.arange(5,31,5),:2];self.reference_states=self.states(adopted)
                self.reference_valid=self.joint_certificate(stop,boxes)['passed'];mode='verified_wait_no_feasible_corridor'
        audit=dict(source='joint_intent_recovery_v12',mode=mode,intent_recovery=self.intent_recovery_audit,geometrically_valid_candidates=sum(c['passed'] for c in getattr(self,'_audit_rows',[])),admitted_candidates=len(candidates),global_stale_veto_used=False,reference_changed=not np.array_equal(raw,adopted),legal_native_reference_preserved=mode=='native_preserved',reference_certificate=self.certificate(self.reference_states),original_certificate=before['road'],selected_candidate=self.selected,tracking_reference_source='dense_kinematic_states' if self.selected is not None else 'native_or_braking',selected_rollout_states=self.selected_states.tolist() if self.selected_states is not None else None,candidates=getattr(self,'_audit_rows',[]),risk_retained=active,stale_risk=stale,motion_evidence=self.motion_evidence,memory=self.memory_audit,stale_policy='per_candidate_geometry_only; no global veto',memory_policy=MEMORY_POLICY,policy=POLICY,seconds=time.perf_counter()-began,initial_edge_alignment_used=False,latch_provenance='current_neural_intent_only; rejected_bypass_has_no_memory_side_effect')
        self._audit_rows=[];self.last_selection=audit
        return adopted,audit

    def observe_conflicts(self,certificate):
        for gap,track in zip(certificate.get('per_actor_minimum_separation_m',[]),self._risk_tracks):
            if gap<.05 and track['last']==self.clock:
                track['latched']=True;track['latch_observation_s']=self.clock;track['latch_source']='current_neural_intent'

    def stop_states(self,motion,action,hold=False,reaction_steps=1):
        m=copy.copy(motion);origin=np.array([motion.z,motion.x]);yaw=motion.yaw;rows=[]
        rate_limit=np.radians(self.v.max_steer_rate_degrees);steer_limit=np.radians(self.v.max_steer_degrees)
        def record():
            dz,dx=np.array([m.z,m.x])-origin;c,s=np.cos(yaw),np.sin(yaw)
            rows.append([c*dx-s*dz-self.mount[0,3],c*dz+s*dx-self.mount[1,3],m.yaw-yaw,m.speed,m.steer])
        def advance(acc,rate):
            acc=float(np.clip(acc,max(self.v.min_acc_mps2,-m.speed/self.v.dt),self.v.max_acc_mps2))
            rate=float(np.clip(rate,max(-rate_limit,(-steer_limit-m.steer)/self.v.dt),min(rate_limit,(steer_limit-m.steer)/self.v.dt)))
            m.advance(dict(acc=acc,steer_rate=rate));record()
        record()
        for _ in range(reaction_steps):advance(action['acc_mps2'],action['steer_rate_radps'])
        for _ in range(int(np.ceil(m.speed/(-self.v.min_acc_mps2*self.v.dt)))+2):
            if m.speed<=1e-9:break
            advance(self.v.min_acc_mps2,0. if hold else -m.steer/self.v.dt)
        return np.asarray(rows)

    def continuation(self,motion,nominal):
        s=np.array([-self.mount[0,3],-self.mount[1,3],0.,motion.speed,motion.steer]);rows=[s.copy()]
        controls=self.selected_controls.copy();controls[0]=[nominal['acc_mps2'],nominal['steer_rate_radps']]
        for acc,rate in controls:
            s[3]=max(0.,s[3]+acc*self.v.dt);s[4]+=rate*self.v.dt
            s[0]+=s[3]*np.sin(s[2])*self.v.dt;s[1]+=s[3]*np.cos(s[2])*self.v.dt
            s[2]+=s[3]*np.tan(s[4])/self.v.wheelbase_m*self.v.dt;rows.append(s.copy())
        return self.terminal_stop(np.array(rows))

    def smooth_action(self,motion,nominal,ceiling=None):
        last=float(motion.last_accel);delta=POLICY['jerk_mps3']*self.v.dt
        physical_low=max(self.v.min_acc_mps2,-motion.speed/self.v.dt)
        physical_high=min(self.v.max_acc_mps2,(self.v.max_speed_mps-motion.speed)/self.v.dt)
        lower=max(physical_low,last-delta)
        upper=min(physical_high,last+delta,physical_high if ceiling is None else ceiling)
        relief=upper<lower-1e-9
        # Speed saturation takes precedence over an infeasible discrete jerk
        # interval. Every resulting action is still jointly verified below.
        if upper<lower:
            point=float(np.clip(ceiling if ceiling is not None else nominal['acc_mps2'],physical_low,physical_high))
            lower=upper=point
        acc=float(np.clip(nominal['acc_mps2'],lower,upper))
        return dict(nominal,acc_mps2=acc),dict(previous_acc_mps2=last,lower_acc_mps2=lower,upper_acc_mps2=upper,jerk_mps3=(acc-last)/self.v.dt,stop_saturation_relief=relief)

    def admit(self,motion,nominal):
        began=time.perf_counter();boxes,tracks=self.actor_boxes()
        original_nominal=nominal.copy()
        ceiling=None if self.selected_states is None else float((self.selected_states[1,3]-motion.speed)/self.v.dt)
        nominal,smoothness=self.smooth_action(motion,nominal,ceiling)
        stop=self.stop_states(motion,nominal,reaction_steps=2);cert=self.joint_certificate(stop,boxes)
        reference=self.actor_certificate(self.reference_states,boxes)
        stale=any(t['latched'] and self.clock-t['last']>POLICY['missing_risk_acceleration_lease_s'] for t in self.tracks)
        continuation=None
        if not cert['passed'] and self.selected is not None:
            continuation=self.joint_certificate(self.continuation(motion,nominal),boxes)
        witness=cert if cert['passed'] else continuation
        allow=self.reference_valid and reference['passed'] and witness is not None and witness['passed'] and self.last_selection['mode']!='verified_wait_no_feasible_corridor'
        audit=dict(execution_smoothness=smoothness,nominal=original_nominal,velocity_envelope_action=nominal,nominal_certificate=cert,continuation_certificate=continuation,reference_actor_certificate=reference,risk_latched=any(t['latched'] for t in self.tracks),actor_evidence=[dict(id=t['id'],age_s=self.clock-t['last'],latched=t['latched']) for t in tracks],stale_risk_present=stale,stale_policy='per_candidate_geometry_only; no global veto',ground_truth_actor_used=False,heavy_mpc_executed=False,selected_candidate=self.selected)
        if allow:
            audit.update(selected='joint_verified_tracker',override=nominal!=original_nominal,certificate=witness,safe_witness='immediate_stop' if cert['passed'] else 'tracking_continuation_with_terminal_stop',seconds=time.perf_counter()-began);return nominal,audit
        cautious=dict(acc_mps2=min(0.,float(motion.last_accel)-POLICY['jerk_mps3']*self.v.dt),steer_rate_radps=float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
        cautious,cautious_smoothness=self.smooth_action(motion,cautious)
        cautious_cert=self.joint_certificate(self.stop_states(motion,cautious,reaction_steps=2),boxes)
        if cautious_cert['passed']:
            audit.update(selected='joint_verified_smooth_protective',override=True,certificate=cautious_cert,execution_smoothness=cautious_smoothness,emergency_jerk_exception=False,protective_phase='braking' if cautious['acc_mps2']<=0 else 'acceleration_unwind',seconds=time.perf_counter()-began)
            return cautious,audit
        audit['cautious_rejection']=cautious_cert
        for hold in (False,True):
            brake=dict(acc_mps2=max(self.v.min_acc_mps2,-motion.speed/self.v.dt),steer_rate_radps=0. if hold else float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
            cert=self.joint_certificate(self.stop_states(motion,brake,hold=hold),boxes)
            if cert['passed']:
                audit.update(emergency_jerk_exception=True,emergency_reason='no_joint_safe_smooth_action',selected='joint_verified_brake_hold' if hold else 'joint_verified_brake_unwind',override=True,certificate=cert,seconds=time.perf_counter()-began);return brake,audit
        audit.update(selected=None,override=True,reason='no_jointly_verified_action',seconds=time.perf_counter()-began);return None,audit
