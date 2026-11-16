"""Small multimodal MPC student; every proposed control requires current constraints."""

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

import hashlib
import json
import time
pass  # Path is provided by the portable boundary above
import numpy as np
import torch
from torch import nn

SCHEMA='offline_mpc_multimodal_student_v1'


class Student(nn.Module):
    def __init__(self,dimension):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(dimension,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.logits=nn.Linear(64,3);self.controls=nn.Linear(64,36)
    def forward(self,x):
        h=self.body(x)
        return self.logits(h),self.controls(h).reshape(-1,3,6,2).tanh()


def control_bounds(vehicle):
    lo=np.tile([vehicle['min_acc_mps2'],-np.deg2rad(vehicle['max_steer_rate_degrees'])],(6,1))
    hi=np.tile([vehicle['max_acc_mps2'],np.deg2rad(vehicle['max_steer_rate_degrees'])],(6,1))
    return lo,hi


def vehicle_hash(vehicle):
    return hashlib.sha256(json.dumps(vehicle,sort_keys=True).encode()).hexdigest()


def family_distances(features,families):
    """Nearest training descriptor from another source frame, never an augmented sibling."""
    squared=((features[:,None,:]-features[None,:,:])**2).mean(-1)
    same=np.array([[a==b for b in families] for a in families])
    squared[same]=np.inf
    result=np.sqrt(squared.min(1))
    if not np.isfinite(result).all():raise ValueError('Need multiple distinct training frame families')
    return result


class Provider:
    def __init__(self,checkpoint,expected_sha256=None,encode=None,vehicle_identity=None):
        path=Path(checkpoint)
        if expected_sha256 and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('Student checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['schema']!=SCHEMA:raise ValueError('Unknown student schema')
        self.state=state;self.model=Student(state['dimension']);self.model.load_state_dict(state['model']);self.model.eval()
        self.mean=np.asarray(state['mean'],dtype=np.float32);self.std=np.asarray(state['std'],dtype=np.float32)
        self.training=np.asarray(state['training_descriptors'],dtype=np.float32)
        if self.training.ndim!=2 or self.training.shape[1]!=state['dimension'] or np.any(self.std<=0):raise ValueError('Invalid student normalization')
        self.encode=encode;self.vehicle_identity=vehicle_identity

    def __call__(self,problem):
        start=time.perf_counter();audit=dict(schema=SCHEMA,student_direct=False,full_current_constraints_checked=False,
            formal_quality_or_safety_certificate=False,ground_truth_used=False)
        def reject(reason):
            audit.update(reason=reason,total_seconds=time.perf_counter()-start)
            return dict(selected=None,audit=audit)
        identity=self.vehicle_identity or (lambda v:vehicle_hash(vars(v)))
        if identity(problem.vehicle)!=self.state['vehicle_identity']:return reject('vehicle_mismatch')
        if self.encode is None:
            from policy_features import encode,FEATURE_SCHEMA
            if FEATURE_SCHEMA!=self.state['feature_schema']:return reject('feature_schema_mismatch')
        else:encode=self.encode
        try:features=np.asarray(encode(problem),dtype=np.float32)
        except (ValueError,FloatingPointError):return reject('invalid_descriptor')
        if features.shape!=(self.state['dimension'],) or not np.isfinite(features).all():return reject('invalid_descriptor')
        normalized=(features-self.mean)/self.std
        distance=float(np.sqrt(np.mean((self.training-normalized)**2,axis=1)).min())
        audit['nearest_training_distance']=distance
        if distance>self.state['distance_threshold']:return reject('outside_training_support')
        with torch.inference_mode():
            logits,controls=self.model(torch.from_numpy(normalized)[None]);prob=logits.softmax(-1)[0].numpy();controls=controls[0].numpy()
        lo,hi=control_bounds(vars(problem.vehicle));controls=lo+(controls+1.)*.5*(hi-lo)
        rows=[];audit['proposal_checks']=[]
        # Only heads with actual teacher labels can propose a control.
        for mode in np.argsort(-prob):
            if int(mode) not in self.state['trained_modes']:continue
            row=problem.evaluate(controls[mode])
            audit['proposal_checks'].append(dict(mode=int(mode),probability=float(prob[mode]),passed=row is not None))
            if row is not None:
                row=dict(row,student_mode=int(mode));rows.append(row)
        audit['full_current_constraints_checked']=True
        if not rows:return reject('no_current_feasible_quality_proposal')
        selected=min(rows,key=lambda x:x['objective'])
        audit.update(reason='checked_student_proposal',student_direct=True,selected_mode=selected['student_mode'],
            objective=selected['objective'],minimum_constraint=selected['minimum_constraint'],
            total_seconds=time.perf_counter()-start,
            objective_optimality_guaranteed=False)
        return dict(selected=selected,audit=audit)


def with_fallback(optimize,provider):
    """Inject only where the existing controller permits a new proposal; retain its fallbacks."""
    def call(reference,occupancy,info,*args,**kwargs):
        if '_proposal_provider' in kwargs:raise ValueError('Cannot replace another proposal provider')
        return optimize(reference,occupancy,info,*args,_proposal_provider=provider,**kwargs)
    return call


def install(driver,checkpoint,expected_sha256=None):
    """Explicit per-process opt-in after phase45 configure; never change disk defaults."""
    location=Path(driver.optimized_controller.__file__).resolve().parent
    if location.name!='framework_driver_v1' or location.parent.name!='phase45':
        raise ValueError('Expected the phase45 driver with current-problem hooks')
    if getattr(driver.base,'_offline_student_installed',False):raise ValueError('Student already installed')
    provider=Provider(checkpoint,expected_sha256)
    original=driver.base.boundary_optimize
    driver.base.boundary_optimize=with_fallback(original,provider)
    driver.base._offline_student_installed=True
    def restore():
        driver.base.boundary_optimize=original;driver.base._offline_student_installed=False
    return provider,restore
