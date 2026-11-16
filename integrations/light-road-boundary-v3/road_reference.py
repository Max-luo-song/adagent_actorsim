"""Current-camera road alignment, model-road cost, full-body stopping certificates.

No scene IDs, source actor truth, route annotations or future ego poses enter here.
The original fixed iLQR is unchanged. This is external reference processing and
a separately declared execution gate, not obstacle MPC or avoidance search.
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

import copy,sys,time
pass  # Path is provided by the portable boundary above
import numpy as np
from common import CR,configure
configure();sys.path.insert(0,str(CR/'phase46/framework_driver_v1'))
from white_edge import detect,WhiteEdge
from road_soft_cost import SoftRoadPenalty
from drivable_area import DrivableArea,from_saved
from planning_guard import sweep_allowance
from tracker import reference

POLICY=dict(raw_weight=1.,observed_road_alignment_weight=25.,smooth_weight=.2,road_cost_weight=10.,
    projected_gradient_iterations=12,finite_difference_m=.01,learning_rate=.015,
    extra_white_line_reserve_m=.05,max_road_heading_slope=.4,
    missing_boundary_policy='no acceleration; brake only with a current full stopping certificate, otherwise no action admitted',
    reference_scope='locally straight continuous observed outer edge; line/ground-plane uncertainty is retained',
    white_line_is_hard_constraint=True,model_road_is_hard_constraint=False,
    model_road_reason='Predicted mask can reject even the current observed on-road footprint; retain as soft evidence, never treat it as a surveyed boundary')

class RoadReference:
    def __init__(self,vehicle):
        self.v=vehicle;self.target_distance=None;self.primary_side=None
        self.edges=None;self.road=None;self.soft=None;self.packet=None;self.mount=None
        self.reference_valid=False

    def states(self,raw):
        states,_=reference(raw,self.mount,self.v.dt)
        return np.c_[states[:,1]-self.mount[0,3],states[:,0]-self.mount[1,3],states[:,2:]]

    def observed(self,obs,info,arrays):
        self.mount=np.asarray(info['model_lidar_frame']['virtual_to_ego'],float)
        self.packet=detect(obs,info);self.edges=WhiteEdge(self.packet,self.v)
        self.road=DrivableArea(from_saved(arrays),self.v);self.soft=SoftRoadPenalty(self.road,self.v)
        return self.packet

    def coverage(self,states):
        """Never treat an unobserved longitudinal band as verified free space."""
        if self.packet is None or self.packet['status']!='fresh':return False
        s=np.asarray(states);hy=abs(np.sin(s[:,2]))*self.v.half_width_m+abs(np.cos(s[:,2]))*self.v.half_length_m
        return any(np.all(s[:,1]-hy>=e['valid_y_range'][0]) and np.all(s[:,1]+hy<=e['valid_y_range'][1]) for e in self.packet['edges'])

    def certificate(self,states):
        s=np.asarray(states);covered=self.coverage(s)
        edge=float(np.min(self.edges.swept_clearance(s))) if self.edges is not None and self.edges.edges else None
        road=float(np.min(self.road.clearance(s)-sweep_allowance(s,self.v))) if self.road is not None else None
        valid=covered and edge is not None and edge>=POLICY['extra_white_line_reserve_m']
        return dict(passed=bool(valid),current_edge_coverage_complete=bool(covered),
            white_edge_swept_clearance_m=edge if edge is not None and np.isfinite(edge) else None,
            predicted_road_swept_clearance_m=road if road is not None and np.isfinite(road) else None,
            predicted_road_is_soft_evidence=True,
            whole_body_and_inter_sample_motion_checked=True,
            physical_white_boundary_certified=False,scope='current camera estimate + uncertainty; not surveyed ground truth')

    def process(self,raw,arrays,obs,info,motion):
        start=time.perf_counter();raw=np.asarray(raw,float);self.reference_valid=False;self.observed(obs,info,arrays)
        eligible=[e for e in self.packet['edges'] if abs(e['a'])<=POLICY['max_road_heading_slope']]
        primary=next((e for e in eligible if e['side']==self.primary_side),None)
        if self.primary_side is None and eligible:
            # Select a physically observed edge, never assume a lane width.
            primary=min(eligible,key=lambda e:abs(e['b']))
            self.primary_side=primary['side']
            origin=-self.mount[:2,3];sign=1 if primary['side']=='right' else -1
            self.target_distance=sign*(primary['a']*origin[1]+primary['b']-origin[0])/np.hypot(1,primary['a'])
        audit=dict(source='camera_road_alignment_and_model_road_cost',candidate_count=1,
            selected='one_projected_reference',rejections=[],patch=True,reference_changed=False,
            observed_white_edge=self.packet,policy=POLICY,ground_truth_used=False,
            online_inputs=['current RGB/calibration','current UniAD map/plan','current ego kinematics'],
            initial_observed_edge_distance_m=self.target_distance,primary_side=self.primary_side)
        if primary is None:
            audit.update(reference_unavailable=True,rejections=['no_current_qualified_primary_edge'],processing_seconds=time.perf_counter()-start)
            return raw.copy(),audit
        sign=1 if primary['side']=='right' else -1
        goal=primary['a']*raw[:,1]+primary['b']-sign*self.target_distance*np.hypot(1,primary['a'])
        D=np.diff(np.eye(6),n=2,axis=0)
        A=(POLICY['raw_weight']+POLICY['observed_road_alignment_weight'])*np.eye(6)+POLICY['smooth_weight']*D.T@D
        rhs=POLICY['raw_weight']*raw[:,0]+POLICY['observed_road_alignment_weight']*goal
        x=np.linalg.solve(A,rhs)
        # Lightweight six-variable projected reference update; no vehicle-control
        # optimization, obstacle constraints, multistart, route or recovery search.
        def road_cost(xx):
            p=raw.copy();p[:,0]=xx
            return self.soft.distance_cost(self.states(p),1)
        def project_band(xx):
            xx=xx.copy()
            for e in self.packet['edges']:
                norm=np.hypot(1,e['a']);extra=self.v.half_width_m+self.v.clearance_margin_m+e['uncertainty_m']+POLICY['extra_white_line_reserve_m']
                boundary=e['a']*raw[:,1]+e['b']
                xx=np.minimum(xx,boundary-extra*norm) if e['side']=='right' else np.maximum(xx,boundary+extra*norm)
            return xx
        def objective(xx):return float(xx@A@xx-2*rhs@xx+POLICY['road_cost_weight']*road_cost(xx))
        x=project_band(x)
        for _ in range(POLICY['projected_gradient_iterations']):
            g=2*(A@x-rhs);eps=POLICY['finite_difference_m']
            for j in range(6):
                delta=np.zeros(6);delta[j]=eps
                g[j]+=POLICY['road_cost_weight']*(road_cost(x+delta)-road_cost(x-delta))/(2*eps)
            step=POLICY['learning_rate'];loss=objective(x)
            for _ in range(8):
                proposal=project_band(x-step*g)
                if objective(proposal)<=loss+1e-10:x=proposal;break
                step*=.5
        adopted=raw.copy();adopted[:,0]=x
        states=self.states(adopted)
        audit.update(reference_changed=not np.allclose(adopted,raw,rtol=0,atol=1e-8),
            raw_road_cost=road_cost(raw[:,0]),adopted_road_cost=road_cost(x),
            raw_road_alignment_error_m=float(np.sqrt(np.mean((raw[:,0]-goal)**2))),
            adopted_road_alignment_error_m=float(np.sqrt(np.mean((x-goal)**2))),
            reference_certificate=self.certificate(states),processing_seconds=time.perf_counter()-start)
        self.reference_valid=audit['reference_certificate']['passed']
        return adopted,audit

    def stopping_rollout(self,motion,action):
        """One proposed native action followed by maximum braking and steering centering."""
        m=copy.copy(motion);origin=np.array([motion.z,motion.x]);yaw=motion.yaw
        rows=[]
        def record():
            dz,dx=np.array([m.z,m.x])-origin;c,s=np.cos(yaw),np.sin(yaw)
            rows.append([c*dx-s*dz-self.mount[0,3],c*dz+s*dx-self.mount[1,3],m.yaw-yaw,m.speed,m.steer])
        record();m.advance(dict(acc=action['acc_mps2'],steer_rate=action['steer_rate_radps']));record()
        maximum=int(np.ceil(m.speed/(-self.v.min_acc_mps2*self.v.dt)))+2
        for _ in range(maximum):
            if m.speed<=1e-9:break
            rate=float(np.clip(-m.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees)))
            m.advance(dict(acc=max(self.v.min_acc_mps2,-m.speed/self.v.dt),steer_rate=rate));record()
        return np.asarray(rows)

    def admit(self,motion,nominal):
        cert=self.certificate(self.stopping_rollout(motion,nominal))
        if cert['passed'] and self.reference_valid:return nominal,dict(selected='fixed_tracker_action',nominal=nominal,certificate=cert,override=False)
        if not self.reference_valid:cert=dict(cert,reference_rejected=True)
        brake=dict(acc_mps2=max(self.v.min_acc_mps2,-motion.speed/self.v.dt),
            steer_rate_radps=float(np.clip(-motion.steer/self.v.dt,-np.radians(self.v.max_steer_rate_degrees),np.radians(self.v.max_steer_rate_degrees))))
        stop=self.certificate(self.stopping_rollout(motion,brake))
        if stop['passed']:return brake,dict(selected='certified_brake',nominal=nominal,nominal_rejection=cert,certificate=stop,override=True)
        return None,dict(selected=None,nominal=nominal,nominal_rejection=cert,brake_rejection=stop,
            override=True,reason='no_verified_road_safe_action; no movement admitted')
